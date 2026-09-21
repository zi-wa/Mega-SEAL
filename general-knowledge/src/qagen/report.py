"""Run summary from raw result files: accuracies, preregistered verdicts, RQ1, compute.

Run as `python -m general-knowledge.src.qagen.report`. summary.json holds every number;
summary.md shows the same numbers as tables. Works on a partial run.
"""
import json
import warnings
from datetime import date

import numpy as np
from scipy.stats import spearmanr

import config

from . import paths

BASELINE = "base_se"
PRIMARY_SEEDS = ("qagen_nN_seed0", "qagen_nN_seed1", "qagen_nN_seed2")
# PREREGISTRATION.md: fixed order, stop at the first failure; H4 is non-inferiority by 3pp
HYPOTHESES = (
    ("H1", "qagen_nN", "base_se", 0.0),
    ("H2", "qagen_n0", "base_se", 0.0),
    ("H3", "qagen_nN", "qagen_n0", 0.0),
    ("H3b", "qagen_nN", "qagen_randR", 0.0),
    ("H4", "qagen_nN", "qa0_sup", -0.03),
)
CORRECTNESS_BAR = 0.8


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def paired_bootstrap(differences, iterations=10_000, seed=0):
    differences = np.asarray(differences, dtype=float)
    draws = np.random.default_rng(seed).integers(0, differences.size, (iterations, differences.size))
    means = differences[draws].mean(axis=1)
    return {"mean": float(differences.mean()), "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5)), "n": int(differences.size)}


def cluster_bootstrap(differences, titles, iterations=10_000, seed=0):
    """Resamples whole SQuAD articles, since passages of one article are not independent."""
    differences = np.asarray(differences, dtype=float)
    groups = [np.flatnonzero(np.asarray(titles) == title) for title in dict.fromkeys(titles)]
    rng = np.random.default_rng(seed)
    means = [differences[np.concatenate([groups[pick] for pick in rng.integers(0, len(groups), len(groups))])].mean()
             for _ in range(iterations)]
    return {"mean": float(differences.mean()), "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5)), "clusters": len(groups)}


def cohens_kappa(first_labels, second_labels):
    first = np.asarray(first_labels, dtype=float)
    second = np.asarray(second_labels, dtype=float)
    observed = float((first == second).mean())
    expected = float(first.mean() * second.mean() + (1 - first.mean()) * (1 - second.mean()))
    if expected == 1.0:
        return float("nan")  # both raters constant: kappa undefined
    return (observed - expected) / (1 - expected)


def title_of(key):
    return key.rsplit("#", 1)[0]


def passage_accuracies(run_dir):
    """condition -> {passage key: accuracy}, plus the per-passage seed mean of the primary condition."""
    tables = {}
    for path in sorted((run_dir / "val").glob("*.json")):
        condition = read_json(path)
        if "passages" in condition:  # val/ also holds the GPT self-edit cache
            tables[path.stem] = {record["key"]: record["accuracy"] for record in condition["passages"]}
    seeds = [tables[name] for name in PRIMARY_SEEDS if name in tables]
    if seeds:
        shared = set.intersection(*(set(seed) for seed in seeds))
        tables["qagen_nN"] = {key: float(np.mean([seed[key] for seed in seeds])) for key in shared}
    return tables


def compare(tables, treatment, control):
    """Paired over passages both conditions measured; None when they share none."""
    shared = sorted(set(tables[treatment]) & set(tables[control]))
    if not shared:
        return None
    differences = [tables[treatment][key] - tables[control][key] for key in shared]
    contrast = paired_bootstrap(differences)
    contrast["title_cluster"] = cluster_bootstrap(differences, [title_of(key) for key in shared])
    return contrast


def hypothesis_verdicts(tables):
    verdicts = []
    for name, treatment, control, bound in HYPOTHESES:
        verdict = {"id": name, "contrast": f"{treatment} - {control} > {bound * 100:+.0f}pp"}
        verdicts.append(verdict)
        if len(verdicts) > 1 and verdicts[-2]["verdict"] != "pass":
            verdict["verdict"] = "not tested"
            continue
        if treatment not in tables or control not in tables:
            verdict["verdict"] = "pending"
            continue
        contrast = compare(tables, treatment, control)
        verdict["result"] = contrast
        passed = contrast is not None and contrast["ci_low"] > bound
        if name == "H1":  # also every training seed on its own must point the same way
            seed_contrasts = [compare(tables, seed, control) for seed in PRIMARY_SEEDS if seed in tables]
            seed_means = [contrast["mean"] for contrast in seed_contrasts if contrast]
            verdict["seed_means"] = seed_means
            passed = passed and len(seed_means) == len(PRIMARY_SEEDS) and min(seed_means) > 0
        verdict["verdict"] = "pass" if passed else "fail"
    return verdicts


def outer_records(run_dir):
    return [record for path in sorted((run_dir / "outer").glob("iter*_records.jsonl"))
            for record in paths.read_jsonl(path)]


def generation_quality(records):
    """RQ1 per outer iteration; failed parses stay in the parse-rate denominator."""
    rows = []
    for iteration in sorted({record["iteration"] for record in records}):
        candidates = [candidate for record in records if record["iteration"] == iteration
                      for candidate in record["candidates"]]
        parsed = [candidate for candidate in candidates if candidate["parse_ok"]]
        row = {"iteration": iteration, "candidates": len(candidates),
               "parse_rate": len(parsed) / len(candidates)}
        for field in ("pair_count", "answer_in_passage", "duplicate_rate", "gold_coverage",
                      "correctness", "closed_book_gen"):
            row[field] = float(np.mean([candidate[field] for candidate in parsed])) if parsed else None
        rows.append(row)
    return rows


def rank_agreement(records, vectors_of):
    """Mean over passages of the Spearman correlation, across self-edits, with the gold accuracy.

    A constant vector has no ranks; such passages are dropped and counted, not scored as zero.
    """
    per_passage, attempted = [], 0
    for record in records:
        vectors = vectors_of(record)
        if not vectors:
            continue
        attempted += 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = [spearmanr(vector, record["qa0_accuracies"]).statistic for vector in vectors]
        values = [value for value in values if not np.isnan(value)]
        if values:
            per_passage.append(float(np.mean(values)))
    return {"bootstrap": paired_bootstrap(per_passage) if per_passage else None,
            "passages": attempted, "dropped": attempted - len(per_passage)}


def proxy_validity(records):
    generated = rank_agreement(
        records, lambda record: [candidate["gen_accuracies"] for candidate in record["candidates"]
                                 if candidate["parse_ok"]])
    null = rank_agreement(
        records, lambda record: [record["null_accuracies"]] if record["null_accuracies"] else [])
    picked_by_generated, picked_by_gold = [], []
    for record in records:
        best_gold = int(np.argmax(record["qa0_accuracies"]))
        for candidate in record["candidates"]:
            if candidate["parse_ok"]:
                positions = range(len(record["qa0_accuracies"]))
                picked_by_generated += [position == candidate["selected"] for position in positions]
                picked_by_gold += [position == best_gold for position in positions]
    kappa = cohens_kappa(picked_by_generated, picked_by_gold) if picked_by_generated else None
    return {"generated_vs_gold": generated, "null_vs_gold": null, "best_self_edit_kappa": kappa}


def rq1_verdicts(validity, records):
    spearman = validity["generated_vs_gold"]["bootstrap"]
    correctness = [candidate["correctness"] for record in records
                   for candidate in record["candidates"] if candidate["parse_ok"]]
    pooled = float(np.mean(correctness)) if correctness else None
    return {
        "spearman_ci_low_above_zero": None if spearman is None else spearman["ci_low"] > 0,
        "correctness_pooled": pooled,
        "correctness_at_least_80pct": None if pooled is None else pooled >= CORRECTNESS_BAR,
    }


def compute_totals(run_dir):
    """TTT count and GPU hours per stage; older files without the fields count as zero."""
    stage_files = {
        "evaluation": [read_json(path) for path in (run_dir / "val").glob("*.json")],
        "se_rl": [read_json(path) for path in (run_dir / "serl").glob("*_round*.json")],
        "outer": [read_json(path) for path in (run_dir / "outer").glob("iter*_summary.json")],
    }
    totals = {
        stage: {"ttt_count": sum(entry.get("ttt_count", 0) for entry in entries),
                "gpu_hours": sum(entry.get("gpu_seconds", 0.0) for entry in entries) / 3600}
        for stage, entries in stage_files.items()
    }
    usage = [read_json(path) for path in sorted((run_dir / "usage").glob("*.json"))]
    totals["grader"] = {field: sum(entry[field] for entry in usage)
                        for field in ("calls", "input_tokens", "output_tokens", "cost_usd")}
    return totals


def build_summary(run_dir):
    tables = passage_accuracies(run_dir)
    records = outer_records(run_dir)
    validity = proxy_validity(records)
    conditions = {
        name: {"accuracy": float(np.mean(list(accuracies.values()))), "n": len(accuracies),
               "vs_base_se": compare(tables, name, BASELINE)
               if BASELINE in tables and name != BASELINE else None}
        for name, accuracies in tables.items()
    }
    return {
        "run": config.RUN_NAME, "model": config.MODEL_NAME, "judge": config.GRADER_MODEL,
        "date": date.today().isoformat(),
        "conditions": conditions,
        "hypotheses": hypothesis_verdicts(tables),
        "generation_quality": generation_quality(records),
        "proxy_validity": validity,
        "rq1_verdicts": rq1_verdicts(validity, records),
        "outer_training": [read_json(path) for path in
                           sorted((run_dir / "outer").glob("iter*_summary.json"))],
        "compute": compute_totals(run_dir),
        "pilot": read_json(run_dir / "dev" / "pilot.json"),
        "judge_validation": read_json(run_dir / "judge" / "validation.json"),
    }


def pp(value):
    return "-" if value is None else f"{value * 100:+.1f}"


def ci(result):
    return "-" if result is None else f"[{result['ci_low'] * 100:+.1f}, {result['ci_high'] * 100:+.1f}]"


def percent(value):
    return "-" if value is None else f"{value * 100:.1f}"


def correlation(entry):
    result = entry["bootstrap"]
    body = "not computable" if result is None else (
        f"{result['mean']:.3f} [{result['ci_low']:.3f}, {result['ci_high']:.3f}] over {result['n']} passages")
    return f"{body}, {entry['dropped']} of {entry['passages']} passages dropped as constant"


def render_markdown(summary):
    lines = [f"# qagen summary: {summary['run']}", "",
             f"- model {summary['model']}, judge {summary['judge']}, {summary['date']}",
             "- accuracies are not comparable to SEAL's published numbers (other model, judge, TTT padding)",
             "", "## Validation accuracy (paired difference vs base_se, 95% bootstrap CI over passages)", "",
             "| condition | accuracy % | n | vs base_se pp | 95% CI pp |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, row in summary["conditions"].items():
        contrast = row["vs_base_se"]
        lines.append(f"| {name} | {percent(row['accuracy'])} | {row['n']} | "
                     f"{pp(contrast and contrast['mean'])} | {ci(contrast)} |")

    lines += ["", "## Preregistered hypotheses (fixed order)", "",
              "| id | contrast | verdict | diff pp | 95% CI pp | title-cluster CI pp |",
              "| --- | --- | --- | ---: | ---: | ---: |"]
    for verdict in summary["hypotheses"]:
        result = verdict.get("result")
        lines.append(f"| {verdict['id']} | {verdict['contrast']} | {verdict['verdict']} | "
                     f"{pp(result and result['mean'])} | {ci(result)} | "
                     f"{ci(result and result['title_cluster'])} |")
    lines.append("- title-cluster CI resamples about 47 articles; small-cluster bootstraps run slightly liberal")

    validity, rq1 = summary["proxy_validity"], summary["rq1_verdicts"]
    kappa = validity["best_self_edit_kappa"]
    lines += ["", "## RQ1: self-generated questions", "",
              f"- Spearman, generated vs gold questions across self-edits: {correlation(validity['generated_vs_gold'])}",
              f"- null reference, another passage's questions: {correlation(validity['null_vs_gold'])}",
              f"- Cohen's kappa, best self-edit by generated vs gold questions: "
              f"{'-' if kappa is None else f'{kappa:.3f}'}",
              f"- criterion Spearman CI lower bound > 0: {rq1['spearman_ci_low_above_zero']}",
              f"- criterion answer correctness >= 80% (pooled over parsed candidates): "
              f"{rq1['correctness_at_least_80pct']} ({percent(rq1['correctness_pooled'])}%)",
              "", "| iteration | candidates | parsed % | pairs | answer in passage % | duplicate % | "
              "gold coverage % | correct % | closed-book % |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["generation_quality"]:
        pairs = "-" if row["pair_count"] is None else f"{row['pair_count']:.1f}"
        lines.append(f"| {row['iteration']} | {row['candidates']} | {percent(row['parse_rate'])} | {pairs} | "
                     f"{percent(row['answer_in_passage'])} | {percent(row['duplicate_rate'])} | "
                     f"{percent(row['gold_coverage'])} | {percent(row['correctness'])} | "
                     f"{percent(row['closed_book_gen'])} |")

    lines += ["", "## Outer loop", "", "| iteration | reward % | full tie % | top tie % | mean margin | kept |",
              "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["outer_training"]:
        lines.append(f"| {row['iteration']} | {percent(row['reward_rate'])} | {percent(row['tie_rate'])} | "
                     f"{percent(row.get('top_tie_rate'))} | {row['mean_margin']:.3f} | {row['kept']} |")

    compute = summary["compute"]
    lines += ["", "## Compute and cost", ""]
    for stage in ("evaluation", "se_rl", "outer"):
        lines.append(f"- {stage}: {compute[stage]['ttt_count']} TTT, {compute[stage]['gpu_hours']:.1f} GPU h")
    grader = compute["grader"]
    lines.append(f"- judge: {grader['calls']} calls, {grader['input_tokens']} in / "
                 f"{grader['output_tokens']} out tokens, {grader['cost_usd']:.2f} USD")
    judge = summary["judge_validation"]
    if judge:
        lines.append(f"- judge vs {judge['cross_model']} on {judge['n']} answers: kappa {judge['kappa']:.3f}, "
                     f"agreement {percent(judge['agreement'])}%, self-flip {percent(judge['self_flip_rate'])}%")
    return "\n".join(lines) + "\n"


def main():
    run_dir = paths.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(run_dir)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(f"wrote {run_dir / 'summary.json'} and {run_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
