"""Outer loop: reward a question set by the self-edit it picks, then finetune the question policy.

Algorithm 1 of the plan. Run as `python -m general-knowledge.src.qagen.run_outer`; re-running
resumes from the per-passage records.
"""
import json
import random
from pathlib import Path
from statistics import mean
from typing import Dict, List

import config

from . import paths, qa_gen, splits
from .grading import Grader
from .inner import (
    adapted_scores,
    closed_book_scores,
    gold_questions,
    question_items,
    sample_self_edits,
)
from .model_ops import Policy


def candidate_questions(passage: Dict, completions: List[str]) -> List[List[Dict[str, str]]]:
    return [qa_gen.parse_qa_pairs(completion) for completion in completions]


def candidate_reward(gen_accuracies: List[float], gold_accuracies: List[float]):
    """The candidate picks the self-edit its questions score best; the reward asks whether that
    pick beats the average self-edit on the gold questions. Self-edits tied at the top share the
    pick, so their gold accuracies are averaged (a set that separates nothing earns nothing)."""
    best = max(gen_accuracies)
    tied = [position for position, accuracy in enumerate(gen_accuracies) if accuracy == best]
    picked_gold = mean(gold_accuracies[position] for position in tied)
    margin = picked_gold - mean(gold_accuracies)
    return tied, margin, int(margin > 0)


def score_passage(policy: Policy, grader: Grader, passage: Dict, iteration: int,
                  passage_index: int, null_pairs: List[Dict[str, str]]) -> Dict:
    """Every candidate picks a self-edit; the reward says whether that pick beats the K-average."""
    ttt_before, seconds_before = policy.ttt_count, policy.ttt_seconds
    seed = iteration * 10_000 + passage_index
    completions = policy.generate(
        [qa_gen.qa_gen_prompt(passage)] * config.QA_CANDIDATES,
        config.QA_GEN_MAX_TOKENS,
        seed=seed,
    )
    candidates = candidate_questions(passage, completions)

    question_sets = {"qa0": gold_questions(passage)}
    for index, pairs in enumerate(candidates):
        if pairs:
            question_sets[f"gen{index}"] = question_items(passage["title"], pairs)
    if null_pairs:
        question_sets["null"] = question_items(passage["title"], null_pairs)

    before = closed_book_scores(policy, grader, question_sets, seed=seed)
    self_edits = sample_self_edits(policy, passage, config.SELF_EDITS, seed=seed + 1)
    ttt_seeds = list(range(config.TTT_SEEDS))
    scores = [
        adapted_scores(policy, grader, passage, self_edit, question_sets, ttt_seeds).means()
        for self_edit in self_edits
    ]
    baseline = before.means()

    gold_accuracies = [score["qa0"] for score in scores]
    gold_answers = [pair["answer"] for pair in passage["questions"]]

    candidate_records = []
    for index, pairs in enumerate(candidates):
        name = f"gen{index}"
        if not pairs:
            candidate_records.append({
                "index": index, "parse_ok": False, "pair_count": 0, "gen_accuracies": [],
                "selected": -1, "tied": [], "margin": 0.0, "reward": 0, "answer_in_passage": 0.0,
                "duplicate_rate": 0.0, "gold_coverage": 0.0, "correctness": 0.0,
                "closed_book_gen": 0.0,
            })
            continue
        gen_accuracies = [score[name] for score in scores]
        tied, margin, reward = candidate_reward(gen_accuracies, gold_accuracies)
        correctness = grader.grade_against_passage(
            [(passage["context"], pair["question"], pair["answer"]) for pair in pairs]
        )
        candidate_records.append({
            "index": index,
            "parse_ok": True,
            "pair_count": len(pairs),
            "gen_accuracies": gen_accuracies,
            "selected": tied[0],
            "tied": tied,
            "margin": margin,
            "reward": reward,
            "answer_in_passage": mean(
                qa_gen.answer_in_passage(pair["answer"], passage["context"]) for pair in pairs
            ),
            "duplicate_rate": qa_gen.duplicate_rate(pairs),
            "gold_coverage": qa_gen.gold_coverage(
                gold_answers, [pair["answer"] for pair in pairs]
            ),
            "correctness": mean(correctness),
            "closed_book_gen": baseline[name],
        })

    rewarded = [record for record in candidate_records if record["reward"] == 1]
    best = max(rewarded, key=lambda record: record["margin"]) if rewarded else None
    return {
        "iteration": iteration,
        "key": splits.passage_key(passage),
        "title": passage["title"],
        "qa_prompt": qa_gen.qa_gen_prompt(passage),
        "closed_book": {"qa0": baseline["qa0"]},
        "qa0_accuracies": gold_accuracies,
        "null_accuracies": [score.get("null", 0.0) for score in scores] if null_pairs else [],
        "candidates": candidate_records,
        "chosen_candidate": best["index"] if best else -1,
        "chosen_completion": completions[best["index"]] if best else "",
        "self_edit_truncated": [
            policy.token_count(self_edit) >= config.SELF_EDIT_MAX_TOKENS for self_edit in self_edits
        ],
        "first_candidate_pairs": next((pairs for pairs in candidates if pairs), []),
        "ttt_count": policy.ttt_count - ttt_before,
        "gpu_seconds": policy.ttt_seconds - seconds_before,
    }


def run_iteration(policy: Policy, grader: Grader, iteration: int, outer_dir: Path) -> List[Dict]:
    records_path = outer_dir / f"iter{iteration}_records.jsonl"
    done = {record["key"]: record for record in paths.read_jsonl(records_path)}

    passages = splits.outer_passages(iteration)
    records = []
    null_pairs: List[Dict[str, str]] = []
    for passage_index, passage in enumerate(passages):
        finished = done.get(splits.passage_key(passage))
        if finished:
            records.append(finished)
            null_pairs = finished.get("first_candidate_pairs", [])
            continue
        record = score_passage(policy, grader, passage, iteration, passage_index, null_pairs)
        null_pairs = record["first_candidate_pairs"]
        paths.append_jsonl(records_path, record)
        records.append(record)
        rewarded = sum(candidate["reward"] for candidate in record["candidates"])
        print(f"[outer {iteration}] {passage_index + 1}/{len(passages)} {passage['title'][:40]} "
              f"rewarded {rewarded}/{config.QA_CANDIDATES}", flush=True)
    return records


def summarize(records: List[Dict], iteration: int, kept: int) -> Dict:
    candidates = [candidate for record in records for candidate in record["candidates"]]
    parsed = [candidate for candidate in candidates if candidate["parse_ok"]]
    ties = [
        candidate for candidate in parsed
        if len(set(candidate["gen_accuracies"])) == 1  # questions cannot separate the self-edits
    ]
    top_ties = [candidate for candidate in parsed if len(candidate.get("tied", [])) > 1]
    return {
        "iteration": iteration,
        "passages": len(records),
        "parse_rate": len(parsed) / len(candidates) if candidates else 0.0,
        "reward_rate": mean(candidate["reward"] for candidate in candidates) if candidates else 0.0,
        "tie_rate": len(ties) / len(parsed) if parsed else 0.0,
        "top_tie_rate": len(top_ties) / len(parsed) if parsed else 0.0,
        "mean_margin": mean(candidate["margin"] for candidate in parsed) if parsed else 0.0,
        "kept": kept,
        "ttt_count": sum(record.get("ttt_count", 0) for record in records),
        "gpu_seconds": sum(record.get("gpu_seconds", 0.0) for record in records),
    }


def run_main_chain(policy: Policy, grader: Grader, outer_dir: Path) -> None:
    for iteration in range(config.OUTER_ITERATIONS):
        summary_path = outer_dir / f"iter{iteration}_summary.json"
        adapter_dir = paths.outer_adapter(iteration)
        if summary_path.exists():
            if adapter_dir.exists():  # an iteration that rewarded nothing left the weights alone
                policy.apply_adapters([str(adapter_dir)])
            continue
        records = run_iteration(policy, grader, iteration, outer_dir)
        pairs = [(record["qa_prompt"], record["chosen_completion"])
                 for record in records if record["chosen_completion"]]
        if pairs:
            policy.finetune_and_merge(pairs, seed=iteration, adapter_dir=str(adapter_dir))
        summary_path.write_text(
            json.dumps(summarize(records, iteration, len(pairs)), indent=2), encoding="utf-8"
        )
        print(f"[outer {iteration}] kept {len(pairs)} question sets", flush=True)


def run_random_chain(policy: Policy, base_weights, outer_dir: Path) -> None:
    """Control chain: same question sampling, but the kept set is drawn at random."""
    policy.restore(base_weights)
    for iteration in range(config.OUTER_ITERATIONS):
        summary_path = outer_dir / f"randr_iter{iteration}_summary.json"
        adapter_dir = paths.outer_adapter(iteration, chain="randr")
        if summary_path.exists():
            if adapter_dir.exists():
                policy.apply_adapters([str(adapter_dir)])
            continue
        main_summary = json.loads(
            (outer_dir / f"iter{iteration}_summary.json").read_text(encoding="utf-8")
        )
        passages = splits.outer_passages(iteration)
        picker = random.Random(iteration)
        chosen_passages = picker.sample(passages, min(main_summary["kept"], len(passages)))
        pairs = []
        for passage_index, passage in enumerate(chosen_passages):
            completions = policy.generate(
                [qa_gen.qa_gen_prompt(passage)] * config.QA_CANDIDATES,
                config.QA_GEN_MAX_TOKENS,
                seed=iteration * 10_000 + passage_index,
            )
            pairs.append((qa_gen.qa_gen_prompt(passage), picker.choice(completions)))
        if pairs:
            policy.finetune_and_merge(pairs, seed=iteration, adapter_dir=str(adapter_dir))
        summary_path.write_text(
            json.dumps({"iteration": iteration, "kept": len(pairs)}, indent=2), encoding="utf-8"
        )
        print(f"[randr {iteration}] kept {len(pairs)} question sets", flush=True)


def main() -> None:
    outer_dir = paths.run_dir() / "outer"
    outer_dir.mkdir(parents=True, exist_ok=True)
    grader = Grader(paths.cache_path())
    policy = Policy.load()
    base_weights = policy.checkpoint()  # exact rollback for the control chain
    run_main_chain(policy, grader, outer_dir)
    run_random_chain(policy, base_weights, outer_dir)
    (paths.run_dir() / "grader_usage_outer.json").write_text(
        json.dumps(grader.usage(), indent=2), encoding="utf-8"
    )
    grader.close()


if __name__ == "__main__":
    main()
