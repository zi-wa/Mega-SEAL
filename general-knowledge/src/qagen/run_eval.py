"""SE-RL stage and held-out evaluation for every condition.

Run as `python -m general-knowledge.src.qagen.run_eval`. Conditions run in priority order and a
finished condition is skipped, so the primary comparison exists even if the machine is stopped.
"""
import json
import zlib
from typing import Callable, Dict, List, Optional

from openai import OpenAI

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
    self_edit_prompt,
)
from .model_ops import Policy

ORDER = [
    "closed_book",
    "passage_only",
    "base_se",
    "qagen_nN_seed0",
    "qagen_n0",
    "qa0_sup",
    "gpt_se",
    "qagen_randR",
    "inner_random",
    "outer_only",
    "closed_book_N",
    "passage_only_N",
    "qagen_nN_seed1",
    "qagen_nN_seed2",
]

# which weights a condition starts from
START = {
    "closed_book": "base", "passage_only": "base", "base_se": "base", "gpt_se": "base",
    "qagen_n0": "base", "qa0_sup": "base", "inner_random": "base",
    "qagen_nN_seed0": "outer", "qagen_nN_seed1": "outer", "qagen_nN_seed2": "outer",
    "outer_only": "outer", "closed_book_N": "outer", "passage_only_N": "outer",
    "qagen_randR": "randr",
}


def start_from(policy: Policy, base_weights, chain: str) -> None:
    policy.restore(base_weights)
    if chain == "base":
        return
    name = "main" if chain == "outer" else "randr"
    adapters = [paths.outer_adapter(iteration, chain=name)
                for iteration in range(config.OUTER_ITERATIONS)]
    policy.apply_adapters([str(adapter) for adapter in adapters if adapter.exists()])


def self_edit_seed(passage: Dict) -> int:
    return zlib.crc32(splits.passage_key(passage).encode("utf-8")) % 100_000


def generated_questions(policy: Policy, passages: List[Dict], questions_key: str) -> Dict[str, list]:
    """One question set per passage from the starting policy, fixed for both ReST-EM rounds."""
    questions_path = paths.run_dir() / "serl" / f"{questions_key}_questions.json"
    if questions_path.exists():
        return json.loads(questions_path.read_text(encoding="utf-8"))
    pairs_by_passage = {}
    for index, passage in enumerate(passages):
        completion = policy.generate(
            [qa_gen.qa_gen_prompt(passage)], config.QA_GEN_MAX_TOKENS, seed=index
        )[0]
        pairs = qa_gen.parse_qa_pairs(completion)
        pairs_by_passage[splits.passage_key(passage)] = pairs
        print(f"[questions {questions_key}] {index + 1}/{len(passages)} {len(pairs)} pairs",
              flush=True)
    questions_path.parent.mkdir(parents=True, exist_ok=True)
    questions_path.write_text(json.dumps(pairs_by_passage, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    return pairs_by_passage


def train_se_rl(policy: Policy, grader: Grader, condition: str,
                reward_questions: Optional[Callable[[Dict], QuestionSet]], seed: int) -> None:
    """Two ReST-EM rounds over the unlabeled passages; a saved round is replayed, not repeated."""
    passages = splits.se_rl_passages()
    serl_dir = paths.run_dir() / "serl"
    serl_dir.mkdir(parents=True, exist_ok=True)
    for round_index in range(config.SE_RL_ROUNDS):
        adapter_dir = paths.se_rl_adapter(f"{condition}_r{round_index}")
        round_path = serl_dir / f"{condition}_round{round_index}.json"
        if round_path.exists():
            if adapter_dir.exists():  # a round with no rewarded pair left the weights alone
                policy.apply_adapters([str(adapter_dir)])
            continue
        if reward_questions is None:
            result = random_selection_round(policy, passages, round_index, seed)
        else:
            result = se_rl_round(policy, grader, passages, reward_questions, round_index, seed,
                                 extra_questions=gold_questions if "qagen" in condition else None)
        if result["pairs"]:
            policy.finetune_and_merge(result["pairs"], seed=seed, adapter_dir=str(adapter_dir))
        round_path.write_text(
            json.dumps({"round": round_index, "passages": result["passages"]},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[se-rl {condition}] round {round_index} trained on {len(result['pairs'])} pairs",
              flush=True)


def gpt_self_edits(passages: List[Dict]) -> Dict[str, List[str]]:
    """SEAL's external-generator baseline: implications written by a GPT model."""
    edits_path = paths.run_dir() / "val" / "gpt_self_edits.json"
    if edits_path.exists():
        return json.loads(edits_path.read_text(encoding="utf-8"))
    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    edits_by_passage = {}
    for index, passage in enumerate(passages):
        prompt = self_edit_prompt(passage)
        completions = []
        for _ in range(config.VAL_SELF_EDITS):
            response = client.responses.create(model=config.GPT_SELF_EDIT_MODEL, input=prompt)
            completions.append(response.output_text)
        edits_by_passage[splits.passage_key(passage)] = completions
        print(f"[gpt self-edits] {index + 1}/{len(passages)}", flush=True)
    edits_path.parent.mkdir(parents=True, exist_ok=True)
    edits_path.write_text(json.dumps(edits_by_passage, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    return edits_by_passage


def prepare(condition: str, policy: Policy, base_weights, grader: Grader) -> Callable:
    """Put the weights in the state this condition needs and return its self-edit source."""
    start_from(policy, base_weights, START[condition])

    if condition in ("closed_book", "closed_book_N"):
        return lambda passage: []
    if condition in ("passage_only", "passage_only_N"):
        return lambda passage: [""]  # train on the passage alone, SEAL's "train on passage only"
    if condition == "gpt_se":
        edits_by_passage = gpt_self_edits(splits.val_passages())
        return lambda passage: edits_by_passage[splits.passage_key(passage)]

    if condition == "qa0_sup":
        train_se_rl(policy, grader, condition, gold_questions, seed=0)
    elif condition == "inner_random":
        train_se_rl(policy, grader, condition, None, seed=0)
    elif condition.startswith("qagen"):
        seed = int(condition[-1]) if condition.startswith("qagen_nN_seed") else 0
        # the three seeds share one question set: only the ReST-EM sampling differs
        questions_key = "qagen_nN" if condition.startswith("qagen_nN") else condition
        pairs_by_passage = generated_questions(policy, splits.se_rl_passages(), questions_key)
        reward = lambda passage: question_items(
            passage["title"], pairs_by_passage[splits.passage_key(passage)]
        )
        train_se_rl(policy, grader, condition, reward, seed=seed)

    return lambda passage: sample_self_edits(
        policy, passage, config.VAL_SELF_EDITS, seed=self_edit_seed(passage)
    )


def main() -> None:
    val_dir = paths.run_dir() / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    grader = Grader(paths.cache_path())
    policy = Policy.load()
    base_weights = policy.checkpoint()
    val_passages = splits.val_passages()

    for condition in ORDER:
        result_path = val_dir / f"{condition}.json"
        if result_path.exists():
            print(f"[eval] {condition} already done", flush=True)
            continue
        self_edit_source = prepare(condition, policy, base_weights, grader)
        result = evaluate_passages(policy, grader, val_passages, self_edit_source, condition)
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        (paths.run_dir() / "grader_usage.json").write_text(
            json.dumps(grader.usage(), indent=2), encoding="utf-8"
        )
        print(f"[eval] {condition} mean {result['mean_accuracy']:.4f}", flush=True)

    grader.close()


if __name__ == "__main__":
    main()
