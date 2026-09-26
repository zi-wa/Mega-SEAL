"""SE-RL stage and held-out evaluation for every run2 condition.

Run as `python -m general-knowledge.src.qagen.run_eval`. Conditions run in the listed order and
a finished condition is skipped, so the run resumes after any stop. Every condition starts from
the base weights; the three treatments share one passage list and one question bank.
"""
import json
import zlib
from typing import Callable, Dict, List, Optional

import config

from . import paths, qa_gen, splits
from .grading import Grader
from .inner import (
    QuestionSet,
    evaluate_passages,
    gold_questions,
    question_items,
    random_selection_round,
    sample_self_edits,
    se_rl_round,
)
from .model_ops import Policy

SEEDS = (0, 1, 2)
TREATMENTS = ("inner_random", "qagen", "qa0_sup")
# seed-major order: a stop after any seed still leaves the three treatments paired
ORDER = ["closed_book", "passage_only", "base_se"] + [
    f"{treatment}_seed{seed}" for seed in SEEDS for treatment in TREATMENTS
]


def self_edit_seed(passage: Dict) -> int:
    return zlib.crc32(splits.passage_key(passage).encode("utf-8")) % 100_000


def question_bank(policy: Policy) -> Dict[str, Dict]:
    """QA_QUESTIONS pairs per passage from the base policy: several completions of the run1
    prompt pooled, deduplicated, cut to the fixed count. Written once, shared by every condition."""
    bank_path = paths.run_dir() / "serl" / "questions.json"
    if bank_path.exists():
        return json.loads(bank_path.read_text(encoding="utf-8"))
    passages = [passage for round_index in range(config.SE_RL_ROUNDS)
                for passage in splits.se_rl_passages(round_index)] + splits.se_rl_reserve()
    bank = {}
    for index, passage in enumerate(passages):
        completions = policy.generate(
            [qa_gen.qa_gen_prompt(passage)] * config.QA_GEN_SAMPLES, config.QA_GEN_MAX_TOKENS, seed=index
        )
        pooled = [pair for completion in completions for pair in qa_gen.parse_qa_pairs(completion)]
        unique = qa_gen.dedup_pairs(pooled)
        bank[splits.passage_key(passage)] = {
            "pairs": unique[: config.QA_QUESTIONS], "parsed": len(pooled), "unique": len(unique),
        }
        print(f"[questions] {index + 1}/{len(passages)} parsed {len(pooled)} unique {len(unique)}",
              flush=True)
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    bank_path.write_text(json.dumps(bank, ensure_ascii=False, indent=2), encoding="utf-8")
    return bank


def fixed_rounds(bank: Dict[str, Dict]) -> List[List[Dict]]:
    """Per round, SE_RL_PASSAGES passages that all hold QA_QUESTIONS pairs; the reserve fills the
    gaps in shuffle order. Saved as keys so every condition and seed trains on the same list."""
    list_path = paths.run_dir() / "serl" / "passages.json"
    by_key = {splits.passage_key(passage): passage
              for round_index in range(config.SE_RL_ROUNDS)
              for passage in splits.se_rl_passages(round_index)}
    by_key.update({splits.passage_key(passage): passage for passage in splits.se_rl_reserve()})
    if list_path.exists():
        keys = json.loads(list_path.read_text(encoding="utf-8"))
        return [[by_key[key] for key in round_keys] for round_keys in keys]

    def full(passage):
        return len(bank[splits.passage_key(passage)]["pairs"]) >= config.QA_QUESTIONS

    reserve = [passage for passage in splits.se_rl_reserve() if full(passage)]
    rounds = []
    for round_index in range(config.SE_RL_ROUNDS):
        block = splits.se_rl_passages(round_index)
        kept = [passage for passage in block if full(passage)]
        missing = config.SE_RL_PASSAGES - len(kept)
        if missing > len(reserve):
            raise RuntimeError(
                f"round {round_index} needs {missing} reserve passages with {config.QA_QUESTIONS} "
                f"questions but only {len(reserve)} are left; raise QA_GEN_SAMPLES or SE_RL_RESERVE "
                f"and delete serl/questions.json")
        kept += reserve[:missing]
        reserve = reserve[missing:]
        rounds.append(kept)
        print(f"[passages] round {round_index}: {len(kept)} passages, "
              f"{sum(passage not in block for passage in kept)} from the reserve", flush=True)
    list_path.write_text(json.dumps([[splits.passage_key(passage) for passage in kept] for kept in rounds],
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    return rounds


def train_se_rl(policy: Policy, grader: Grader, condition: str,
                reward_questions: Optional[Callable[[Dict], QuestionSet]], seed: int,
                rounds: List[List[Dict]]) -> None:
    """ReST-EM rounds over the fixed passage lists; a saved round is replayed, not repeated."""
    serl_dir = paths.run_dir() / "serl"
    for round_index, passages in enumerate(rounds):
        adapter_dir = paths.se_rl_adapter(f"{condition}_r{round_index}")
        round_path = serl_dir / f"{condition}_round{round_index}.json"
        if round_path.exists():
            saved = json.loads(round_path.read_text(encoding="utf-8"))
            if saved["pair_count"] and not adapter_dir.exists():
                raise RuntimeError(f"{round_path.name} trained {saved['pair_count']} pairs but "
                                   f"{adapter_dir} is missing; the resume would run on the wrong weights")
            if adapter_dir.exists():  # a round with no pair left the weights alone
                policy.apply_adapters([str(adapter_dir)])
            continue
        progress_path = serl_dir / f"{condition}_round{round_index}.progress.jsonl"
        if reward_questions is None:
            result = random_selection_round(policy, passages, round_index, seed)
        else:
            result = se_rl_round(policy, grader, passages, reward_questions, round_index, seed,
                                 progress_path,
                                 extra_questions=gold_questions if condition.startswith("qagen") else None)
        if result["pairs"]:
            policy.finetune_and_merge(result["pairs"], seed=seed, adapter_dir=str(adapter_dir))
        round_path.write_text(
            json.dumps({"round": round_index, "passages": result["passages"],
                        "pair_count": len(result["pairs"]),
                        "ttt_count": result["ttt_count"], "gpu_seconds": result["gpu_seconds"]},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        progress_path.unlink(missing_ok=True)
        print(f"[se-rl {condition}] round {round_index} trained on {len(result['pairs'])} pairs",
              flush=True)


def prepare(condition: str, policy: Policy, base_weights, grader: Grader,
            rounds: List[List[Dict]], bank: Dict[str, Dict]) -> Callable:
    """Put the weights in the state this condition needs and return its self-edit source."""
    policy.restore(base_weights)
    if condition == "closed_book":
        return lambda passage: []
    if condition == "passage_only":
        # train on the passage alone, SEAL's "train on passage only", as many TTTs as self-edits
        return lambda passage: [""] * config.VAL_SELF_EDITS
    if condition != "base_se":
        treatment, seed = condition.rsplit("_seed", 1)
        if treatment == "qa0_sup":
            reward = gold_questions
        elif treatment == "inner_random":
            reward = None
        else:
            # exactly QA_QUESTIONS per passage, also if the bank was cut at a larger setting
            reward = lambda passage: question_items(
                passage["title"], bank[splits.passage_key(passage)]["pairs"][: config.QA_QUESTIONS]
            )
        train_se_rl(policy, grader, condition, reward, int(seed), rounds)
    return lambda passage: sample_self_edits(
        policy, passage, config.VAL_SELF_EDITS, seed=self_edit_seed(passage)
    )


def main() -> None:
    val_dir = paths.run_dir() / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    (paths.run_dir() / "serl").mkdir(parents=True, exist_ok=True)
    grader = Grader(paths.cache_path())
    policy = Policy.load()
    base_weights = policy.checkpoint()
    bank = question_bank(policy)
    rounds = fixed_rounds(bank)
    val_passages = splits.val_passages()

    for condition in ORDER:
        result_path = val_dir / f"{condition}.json"
        if result_path.exists():
            print(f"[eval] {condition} already done", flush=True)
            continue
        self_edit_source = prepare(condition, policy, base_weights, grader, rounds, bank)
        progress_path = val_dir / f"{condition}.progress.jsonl"
        result = evaluate_passages(policy, grader, val_passages, self_edit_source, condition,
                                   progress_path)
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        progress_path.unlink(missing_ok=True)
        grader.save_usage(paths.usage_path("eval"))
        print(f"[eval] {condition} mean {result['mean_accuracy']:.4f}", flush=True)

    grader.close()


if __name__ == "__main__":
    main()
