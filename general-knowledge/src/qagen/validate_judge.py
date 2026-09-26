"""Judge check before the main run: agreement with SEAL's judge, and self-consistency.

Run as `python -m general-knowledge.src.qagen.validate_judge` after the pilot. The judge is frozen
after this; swapping it later would invalidate every cached verdict.
"""
import json
import random

import config

from . import paths
from .grading import Grader


def cohens_kappa(first_labels, second_labels):
    agreement = sum(a == b for a, b in zip(first_labels, second_labels)) / len(first_labels)
    first_yes = sum(first_labels) / len(first_labels)
    second_yes = sum(second_labels) / len(second_labels)
    chance = first_yes * second_yes + (1 - first_yes) * (1 - second_yes)
    return float("nan") if chance == 1 else (agreement - chance) / (1 - chance)


def main() -> None:
    samples_path = paths.run_dir() / "dev" / "judge_samples.json"
    if not samples_path.exists():
        raise FileNotFoundError(f"run the pilot first: {samples_path} is missing")
    judge_dir = paths.run_dir() / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    validation_path = judge_dir / "validation.json"
    if validation_path.exists():
        print("[judge] already validated", flush=True)
        return

    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    picker = random.Random(0)
    cross_sample = picker.sample(samples, min(config.CROSS_GRADER_SAMPLE, len(samples)))
    repeat_sample = picker.sample(samples, min(config.SELF_CONSISTENCY_SAMPLE, len(samples)))

    items = [(sample["question"], sample["gold"], sample["prediction"]) for sample in cross_sample]
    cross_grader = Grader(
        judge_dir / "cross_cache.jsonl",
        model=config.CROSS_GRADER_MODEL,
        reasoning_effort="",  # SEAL graded with plain greedy decoding
    )
    cross_verdicts = cross_grader.grade(items)
    cross_grader.close()

    run_grader = Grader(paths.cache_path())
    repeat_verdicts = run_grader.grade_uncached(
        [(sample["question"], sample["gold"], sample["prediction"]) for sample in repeat_sample]
    )
    run_grader.save_usage(paths.usage_path("judge_check"))
    run_grader.close()

    run_verdicts = [sample["verdict"] for sample in cross_sample]
    agreement = sum(a == b for a, b in zip(run_verdicts, cross_verdicts)) / len(cross_sample)
    flips = sum(sample["verdict"] != verdict
                for sample, verdict in zip(repeat_sample, repeat_verdicts)) / len(repeat_sample)

    validation = {
        "judge_model": config.GRADER_MODEL,
        "cross_model": config.CROSS_GRADER_MODEL,
        "n": len(cross_sample),
        "kappa": cohens_kappa(run_verdicts, cross_verdicts),
        "agreement": agreement,
        "self_flip_rate": flips,
        "self_consistency_n": len(repeat_sample),
        "cross_grader_usage": cross_grader.usage(),
    }
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(f"[judge] kappa {validation['kappa']:.3f} agreement {agreement:.3f} "
          f"self-flip {flips:.3f}", flush=True)


if __name__ == "__main__":
    main()
