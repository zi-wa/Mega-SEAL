"""Dev pilot: measure TTT cost, check the inner loop helps at all, collect judge samples.

Run as `python -m general-knowledge.src.qagen.run_pilot`. Uses the dev passages only; the
validation passages stay untouched until the final evaluation.
"""
import json
import time
from typing import Dict, List

import config

from . import paths, qa_gen, splits
from .grading import Grader
from .inner import (
    adapted_scores,
    evaluate_passages,
    gold_questions,
    question_items,
    sample_self_edits,
)
from .model_ops import Policy

SAMPLE_PASSAGES = 10  # judge-validation answers come from this many dev passages


def projected_hours(cycle_seconds: float) -> Dict[str, float]:
    outer = config.OUTER_ITERATIONS * config.OUTER_PASSAGES * config.SELF_EDITS * config.TTT_SEEDS
    se_rl_runs = 6  # qa0_sup, qagen_n0, qagen_nN x 3 seeds, qagen_randR
    se_rl = (se_rl_runs * config.SE_RL_ROUNDS * config.SE_RL_PASSAGES
             * config.SELF_EDITS * config.TTT_SEEDS)
    val_conditions = 12  # every condition whose evaluation adapts, see run_eval.ORDER
    validation = val_conditions * config.VAL_PASSAGES * config.VAL_SELF_EDITS
    hours = {
        "outer": outer * cycle_seconds / 3600,
        "se_rl": se_rl * cycle_seconds / 3600,
        "validation": validation * cycle_seconds / 3600,
    }
    hours["total_ttt"] = sum(hours.values())
    hours["ttt_count"] = outer + se_rl + validation
    return hours


def judge_samples(policy: Policy, grader: Grader, passages: List[Dict]) -> List[Dict]:
    """Answers with their verdicts, so the judge can be cross-checked before the main run."""
    samples = []
    for index, passage in enumerate(passages):
        questions = gold_questions(passage)
        self_edit = sample_self_edits(policy, passage, 1, seed=index)[0]
        generated = qa_gen.parse_qa_pairs(
            policy.generate([qa_gen.qa_gen_prompt(passage)], config.QA_GEN_MAX_TOKENS, seed=index)[0]
        )
        question_sets = {"gold": questions}
        if generated:
            question_sets["generated"] = question_items(passage["title"], generated)
        pending = adapted_scores(policy, grader, passage, self_edit, question_sets, [0])
        for name, items in question_sets.items():
            verdicts = [future.result() for future in pending.per_seed[0][name]]
            predictions = pending.answers[0][name]
            for item, prediction, verdict in zip(items, predictions, verdicts):
                samples.append({
                    "set": name,
                    "question": item["question"],
                    "gold": item["answer"],
                    "prediction": prediction,
                    "verdict": verdict,
                })
        print(f"[pilot samples] {index + 1}/{len(passages)}", flush=True)
    return samples


def main() -> None:
    dev_dir = paths.run_dir() / "dev"
    dev_dir.mkdir(parents=True, exist_ok=True)
    pilot_path = dev_dir / "pilot.json"
    if pilot_path.exists():
        print("[pilot] already done", flush=True)
        return

    grader = Grader(paths.cache_path())
    policy = Policy.load()
    passages = splits.dev_passages()
    one_self_edit = lambda passage: sample_self_edits(policy, passage, 1, seed=len(passage["context"]))

    conditions = {}
    for label, source in [("closed_book", lambda passage: []),
                          ("passage_only", lambda passage: [""]),
                          ("base_se", one_self_edit)]:
        started, ttt_before = time.time(), policy.ttt_count
        measured = evaluate_passages(policy, grader, passages, source, f"dev_{label}",
                                     dev_dir / f"{label}.progress.jsonl")
        conditions[label] = measured["mean_accuracy"]
        if label == "base_se":
            ttt_seconds = measured["gpu_seconds"] / max(measured["ttt_count"], 1)
            # self-edit sampling, LoRA training, answers and judge waits per adaptation
            cycle_seconds = (time.time() - started) / max(policy.ttt_count - ttt_before, 1)

    config.SEAL_PAD_QUIRK = True  # ablation: SEAL pads to 2048 with eos and trains on the padding
    padded = evaluate_passages(policy, grader, passages, one_self_edit, "dev_base_se_seal_padding",
                               dev_dir / "base_se_seal_padding.progress.jsonl")
    config.SEAL_PAD_QUIRK = False
    conditions["base_se_seal_padding"] = padded["mean_accuracy"]

    samples = judge_samples(policy, grader, passages[:SAMPLE_PASSAGES])
    (dev_dir / "judge_samples.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    hours = projected_hours(cycle_seconds)
    pilot_path.write_text(
        json.dumps({
            "ttt_seconds": ttt_seconds,
            "cycle_seconds": cycle_seconds,
            "conditions": conditions,
            "projection_hours": hours,
        }, indent=2),
        encoding="utf-8",
    )
    grader.save_usage(paths.usage_path("pilot"))
    for progress_path in dev_dir.glob("*.progress.jsonl"):
        progress_path.unlink()  # pilot.json now marks the pilot finished
    grader.close()

    print(f"[pilot] one adaptation cycle takes {cycle_seconds:.1f}s "
          f"(LoRA training alone {ttt_seconds:.1f}s)", flush=True)
    print(f"[pilot] closed book {conditions['closed_book']:.3f} | "
          f"passage only {conditions['passage_only']:.3f} | "
          f"self-edit {conditions['base_se']:.3f} | "
          f"self-edit with SEAL padding {conditions['base_se_seal_padding']:.3f}", flush=True)
    print(f"[pilot] projected hours: outer {hours['outer']:.1f}, se-rl {hours['se_rl']:.1f}, "
          f"validation {hours['validation']:.1f}, total {hours['total_ttt']:.1f} "
          f"over {int(hours['ttt_count'])} adaptations", flush=True)


if __name__ == "__main__":
    main()
