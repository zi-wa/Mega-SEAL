# general-knowledge/src/qagen/report.py
"""
Aggregate the raw qagen run files into summary.json and summary.md.
Usage:
    python -m general-knowledge.src.qagen.report
"""
import json
from datetime import date

import numpy as np
from scipy.stats import spearmanr

import config

from . import paths

REPORT_ORDER = (
    "closed_book",
    "passage_only",
    "base_se",
    "gpt_se",
    "qa0_sup",
    "qagen_n0",
    "qagen_nN_seed0",
    "qagen_nN_seed1",
    "qagen_nN_seed2",
    "qagen_nN",
    "qagen_randR",
    "inner_random",
    "outer_only",
    "closed_book_N",
    "passage_only_N",
)
SEED_CONDITIONS = ("qagen_nN_seed0", "qagen_nN_seed1", "qagen_nN_seed2")
SEED_MEAN = "qagen_nN"
NON_INFERIORITY_MARGIN = 0.03
HEADER_NOTE = (
    "Absolute accuracies are not comparable to SEAL's published numbers: "
    "different model, judge and TTT padding."
)

HYPOTHESES = (
    ("H1", "qagen_nN > base_se", SEED_MEAN, "base_se", 0.0),
    ("H2", "qagen_n0 > base_se", "qagen_n0", "base_se", 0.0),
    ("H3", "qagen_nN > qagen_n0", SEED_MEAN, "qagen_n0", 0.0),
    ("H3b", "qagen_nN > qagen_randR", SEED_MEAN, "qagen_randR", 0.0),
    ("H4", "qagen_nN >= qa0_sup - 3pp", SEED_MEAN, "qa0_sup", -NON_INFERIORITY_MARGIN),
)


def paired_bootstrap(paired_differences, iterations=10000, seed=0):
    differences = np.asarray(paired_differences, dtype=float)
    draws = np.random.default_rng(seed).integers(0, differences.size, size=(iterations, differences.size))
    resampled_means = differences[draws].mean(axis=1)
    return {
        "mean": float(differences.mean()),
        "ci_low": float(np.percentile(resampled_means, 2.5)),
        "ci_high": float(np.percentile(resampled_means, 97.5)),
        "n": int(differences.size),
    }


def cluster_bootstrap(paired_differences, cluster_titles, iterations=10000, seed=0):
    # passages of one SQuAD article are not independent, so articles are resampled with all their passages
    differences = np.asarray(paired_differences, dtype=float)
    if differences.size == 0:
        return None  # no passages shared between the two conditions, nothing to cluster
    title_column = np.asarray(cluster_titles)
    titles = list(dict.fromkeys(cluster_titles))
    rows_by_title = [np.flatnonzero(title_column == title) for title in titles]
    rng = np.random.default_rng(seed)
    resampled_means = np.empty(iterations)
    for iteration in range(iterations):
        picked = rng.integers(0, len(titles), size=len(titles))
        resampled_means[iteration] = differences[np.concatenate([rows_by_title[index] for index in picked])].mean()
    return {
        "mean": float(differences.mean()),
        "ci_low": float(np.percentile(resampled_means, 2.5)),
        "ci_high": float(np.percentile(resampled_means, 97.5)),
        "clusters": len(titles),
    }


def cohens_kappa(first_labels, second_labels):
    first = np.asarray(first_labels, dtype=float)
    second = np.asarray(second_labels, dtype=float)
    observed = float((first == second).mean())
    expected = float(first.mean() * second.mean() + (1 - first.mean()) * (1 - second.mean()))
    if expected == 1.0:
        return float("nan")  # both judges said the same thing to everything; kappa is undefined
    return (observed - expected) / (1 - expected)


def read_optional(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_conditions(run_dir):
    conditions = {}
    for name in REPORT_ORDER:
        condition = read_optional(run_dir / "val" / f"{name}.json")
        if condition is not None:
            conditions[name] = condition
    return conditions


def accuracy_by_passage(condition):
    # key is title + context hash (unique per passage), so pairing does not depend on file order
    return {passage["key"]: passage["accuracy"] for passage in condition["passages"]}


def title_by_key(condition):
    return {
        passage["key"]: passage.get("title", passage["key"].rsplit("#", 1)[0])
        for passage in condition["passages"]
    }


def paired_differences(treatment_accuracies, baseline_accuracies):
    shared = [key for key in treatment_accuracies if key in baseline_accuracies]
    differences = [treatment_accuracies[key] - baseline_accuracies[key] for key in shared]
    return differences, shared


def seed_mean_accuracies(accuracy_tables):
    seed_tables = [accuracy_tables[name] for name in SEED_CONDITIONS if name in accuracy_tables]
    if not seed_tables:
        return {}
    return {
        key: float(np.mean([table[key] for table in seed_tables]))
        for key in seed_tables[0]
        if all(key in table for table in seed_tables)
    }


def condition_rows(conditions, accuracy_tables, title_lookup):
    baseline = accuracy_tables.get("base_se")
    seed_files = [conditions[name] for name in SEED_CONDITIONS if name in conditions]
    rows = []
    for name in REPORT_ORDER:
        if name not in accuracy_tables:
            continue
        sources = [conditions[name]] if name in conditions else seed_files
        row = {
            "condition": name,
            "mean_accuracy": float(np.mean([source["mean_accuracy"] for source in sources])),
            "passages": len(conditions[name]["passages"]) if name in conditions else len(accuracy_tables[name]),
            "ttt_count": float(np.mean([source.get("ttt_count", 0) for source in sources])),
            "gpu_hours": float(np.mean([source.get("gpu_seconds", 0) for source in sources])) / 3600.0,
            "difference": None,
            "cluster_difference": None,
            "cluster_difference_reason": None,
        }
        if baseline is not None and name != "base_se":
            differences, keys = paired_differences(accuracy_tables[name], baseline)
            row["difference"] = paired_bootstrap(differences)
            if name == SEED_MEAN:
                titles = [title_lookup[key] for key in keys]
                row["cluster_difference"] = cluster_bootstrap(differences, titles)
                if row["cluster_difference"] is None:
                    row["cluster_difference_reason"] = "no passages shared with base_se"
        rows.append(row)
    return rows


def run_hypothesis_sequence(accuracy_tables, title_lookup):
    verdicts = []
    stopped = False
    for identifier, statement, treatment, baseline, threshold in HYPOTHESES:
        verdict = {
            "id": identifier,
            "statement": statement,
            "verdict": "not tested",
            "reason": None,
            "bootstrap": None,
            "cluster_bootstrap": None,
            "seed_differences": None,
        }
        if stopped:
            verdicts.append(verdict)
            continue
        missing = [name for name in (treatment, baseline) if name not in accuracy_tables]
        if missing:
            verdict["reason"] = "missing " + ", ".join(missing)
            verdicts.append(verdict)
            stopped = True
            continue
        differences, keys = paired_differences(accuracy_tables[treatment], accuracy_tables[baseline])
        verdict["bootstrap"] = paired_bootstrap(differences)
        passed = verdict["bootstrap"]["ci_low"] > threshold
        if identifier == "H1":
            titles = [title_lookup[key] for key in keys]
            verdict["cluster_bootstrap"] = cluster_bootstrap(differences, titles)
            seed_differences = {
                name: float(np.mean(paired_differences(accuracy_tables[name], accuracy_tables[baseline])[0]))
                for name in SEED_CONDITIONS
                if name in accuracy_tables
            }
            verdict["seed_differences"] = seed_differences
            all_seeds = len(seed_differences) == len(SEED_CONDITIONS)
            same_sign = all_seeds and all(difference > 0 for difference in seed_differences.values())
            if not all_seeds:
                verdict["reason"] = "fewer than three seeds present"
            elif not same_sign:
                verdict["reason"] = "seed differences not all positive"
            passed = passed and same_sign
        verdict["verdict"] = "pass" if passed else "fail"
        stopped = not passed
        verdicts.append(verdict)
    return verdicts


def load_outer_summaries(run_dir):
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in (run_dir / "outer").glob("iter*_summary.json")]
    return sorted(summaries, key=lambda summary: summary["iteration"])


def load_outer_records(run_dir):
    records = []
    for path in (run_dir / "outer").glob("iter*_records.jsonl"):
        with path.open(encoding="utf-8") as lines:
            records.extend(json.loads(line) for line in lines if line.strip())
    return records


def candidate_mean(candidates, field):
    if not candidates:
        return float("nan")
    return float(np.mean([candidate[field] for candidate in candidates]))


def generation_quality(records):
    candidates_by_iteration = {}
    for record in records:
        candidates_by_iteration.setdefault(record["iteration"], []).extend(record["candidates"])
    rows = []
    for iteration in sorted(candidates_by_iteration):
        candidates = candidates_by_iteration[iteration]
        # unparsed candidates carry no questions, so the quality means run over the parsed ones only
        parsed = [candidate for candidate in candidates if candidate["parse_ok"]]
        rows.append({
            "iteration": iteration,
            "candidates": len(candidates),
            "parse_rate": float(np.mean([float(candidate["parse_ok"]) for candidate in candidates])),
            "pairs_per_candidate": candidate_mean(parsed, "pair_count"),
            "answer_in_passage": candidate_mean(parsed, "answer_in_passage"),
            "duplicate_rate": candidate_mean(parsed, "duplicate_rate"),
            "gold_coverage": candidate_mean(parsed, "gold_coverage"),
            "correctness": candidate_mean(parsed, "correctness"),
            "closed_book_gen": candidate_mean(parsed, "closed_book_gen"),
        })
    return rows


def proxy_validity(records):
    generated_correlations = []
    generated_attempted = 0
    generated_dropped = 0
    null_correlations = []
    null_attempted = 0
    null_dropped = 0
    selected_labels = []
    gold_labels = []
    for record in records:
        gold_accuracies = record["qa0_accuracies"]
        best_by_gold = int(np.argmax(gold_accuracies))
        parsed_candidates = [candidate for candidate in record["candidates"] if candidate["parse_ok"]]
        passage_correlations = []
        for candidate in parsed_candidates:
            correlation = float(spearmanr(candidate["gen_accuracies"], gold_accuracies).statistic)
            # a self-edit set scored identically by every question gives no ranks to correlate
            if not np.isnan(correlation):
                passage_correlations.append(correlation)
            selected_labels.extend(
                1 if index == candidate["selected"] else 0 for index in range(len(gold_accuracies))
            )
            gold_labels.extend(1 if index == best_by_gold else 0 for index in range(len(gold_accuracies)))
        if parsed_candidates:
            generated_attempted += 1
            if passage_correlations:
                generated_correlations.append(float(np.mean(passage_correlations)))
            else:
                generated_dropped += 1
        # the first passage of an iteration has no earlier passage to borrow a null question set from
        if len(record["null_accuracies"]) == len(gold_accuracies):
            null_attempted += 1
            null_correlation = float(spearmanr(record["null_accuracies"], gold_accuracies).statistic)
            if not np.isnan(null_correlation):
                null_correlations.append(null_correlation)
            else:
                null_dropped += 1
    return {
        "generated_vs_gold": paired_bootstrap(generated_correlations) if generated_correlations else None,
        "generated_attempted": generated_attempted,
        "generated_dropped": generated_dropped,
        "null_vs_gold": paired_bootstrap(null_correlations) if null_correlations else None,
        "null_attempted": null_attempted,
        "null_dropped": null_dropped,
        "best_self_edit_kappa": cohens_kappa(selected_labels, gold_labels) if selected_labels else None,
        "kappa_items": len(selected_labels),
    }


def qa_gen_correctness(records):
    # pooled mean over every parsed candidate across all outer iterations, not the mean of per-iteration means
    parsed = [candidate for record in records for candidate in record["candidates"] if candidate["parse_ok"]]
    return candidate_mean(parsed, "correctness")


def rq1_verdicts(validity, records):
    generated = validity["generated_vs_gold"]
    if generated is None:
        spearman_verdict = {
            "id": "RQ1-spearman",
            "statement": "mean within-passage Spearman (generated vs gold) bootstrap 95% CI lower bound > 0",
            "verdict": "not tested",
            "reason": "outer records not present",
            "ci_low": None,
        }
    else:
        spearman_verdict = {
            "id": "RQ1-spearman",
            "statement": "mean within-passage Spearman (generated vs gold) bootstrap 95% CI lower bound > 0",
            "verdict": "pass" if generated["ci_low"] > 0 else "fail",
            "reason": None,
            "ci_low": generated["ci_low"],
        }
    if not records:
        correctness_verdict = {
            "id": "RQ1-correctness",
            "statement": "QA_gen answer correctness >= 80%, pooled mean of correctness over parsed "
                          "candidates across all outer iterations",
            "verdict": "not tested",
            "reason": "outer records not present",
            "mean_correctness": None,
        }
    else:
        correctness = qa_gen_correctness(records)
        correctness_verdict = {
            "id": "RQ1-correctness",
            "statement": "QA_gen answer correctness >= 80%, pooled mean of correctness over parsed "
                          "candidates across all outer iterations",
            "verdict": "pass" if not np.isnan(correctness) and correctness >= 0.8 else "fail",
            "reason": None,
            "mean_correctness": None if np.isnan(correctness) else correctness,
        }
    return [spearman_verdict, correctness_verdict]


def percent(value):
    if value is None or np.isnan(value):
        return "n/a"
    return f"{value * 100:.1f}"


def points(value):
    if value is None or np.isnan(value):
        return "n/a"
    return f"{value * 100:+.1f}"


def interval_points(bootstrap):
    if bootstrap is None:
        return "-"
    return f"[{bootstrap['ci_low'] * 100:+.1f}, {bootstrap['ci_high'] * 100:+.1f}]"


def render_conditions(rows):
    lines = [
        "| condition | accuracy % | n | diff vs base_se (pp) | 95% CI (pp) | title-cluster 95% CI (pp) | TTT | GPU h |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        difference = "-" if row["difference"] is None else points(row["difference"]["mean"])
        lines.append(
            f"| {row['condition']} | {percent(row['mean_accuracy'])} | {row['passages']} | "
            f"{difference} | {interval_points(row['difference'])} | {interval_points(row['cluster_difference'])} | "
            f"{row['ttt_count']:.0f} | {row['gpu_hours']:.2f} |"
        )
    return lines


def render_hypotheses(verdicts):
    lines = [
        "| id | statement | verdict | mean diff (pp) | 95% CI (pp) | note |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for verdict in verdicts:
        bootstrap = verdict["bootstrap"]
        difference = "-" if bootstrap is None else points(bootstrap["mean"])
        notes = [verdict["reason"]] if verdict["reason"] else []
        if verdict["seed_differences"]:
            seeds = ", ".join(f"{name}: {points(value)}" for name, value in verdict["seed_differences"].items())
            notes.append(f"seeds {seeds}")
        if verdict["cluster_bootstrap"]:
            notes.append(f"title-cluster CI {interval_points(verdict['cluster_bootstrap'])}")
        lines.append(
            f"| {verdict['id']} | {verdict['statement']} | {verdict['verdict']} | "
            f"{difference} | {interval_points(bootstrap)} | {'; '.join(notes)} |"
        )
    return lines


def render_markdown(summary):
    run = summary["run"]
    lines = [
        f"# qagen summary: {run['name']}",
        "",
        f"- run: {run['name']}",
        f"- model: {run['model']}",
        f"- judge: {run['judge_model']}",
        f"- date: {run['date']}",
        f"- {run['note']}",
        "",
        "## Conditions",
        "",
    ]
    lines += render_conditions(summary["conditions"])
    lines += [
        "",
        "Title-cluster bootstrap resamples about 47 SQuAD article clusters; "
        "small-cluster bootstraps run somewhat liberal.",
    ]
    lines += ["", "## Hypotheses, fixed sequence", ""]
    lines += render_hypotheses(summary["hypotheses"])

    lines += ["", "## RQ1 generation quality per outer iteration", ""]
    quality_rows = summary["generation_quality"]
    if not quality_rows:
        lines.append("outer records not present")
    else:
        lines += [
            "| iteration | candidates | parse % | pairs/candidate | answer in passage % | duplicate % | "
            "gold coverage % | QA_gen correctness % | closed-book on generated Q % |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in quality_rows:
            lines.append(
                f"| {row['iteration']} | {row['candidates']} | {percent(row['parse_rate'])} | "
                f"{row['pairs_per_candidate']:.2f} | {percent(row['answer_in_passage'])} | "
                f"{percent(row['duplicate_rate'])} | {percent(row['gold_coverage'])} | "
                f"{percent(row['correctness'])} | {percent(row['closed_book_gen'])} |"
            )

    lines += ["", "## RQ1 proxy validity", ""]
    validity = summary["proxy_validity"]
    generated = validity["generated_vs_gold"]
    null_reference = validity["null_vs_gold"]
    if generated is None:
        lines.append("outer records not present")
    else:
        lines += [
            f"- mean within-passage Spearman, generated vs gold questions: {generated['mean']:.3f} "
            f"[{generated['ci_low']:.3f}, {generated['ci_high']:.3f}] over {generated['n']} passages",
            f"- generated-question correlation dropped {validity['generated_dropped']} of "
            f"{validity['generated_attempted']} passages (constant accuracy vector, undefined Spearman)",
            f"- null reference, other passage's questions vs gold: {null_reference['mean']:.3f} "
            f"[{null_reference['ci_low']:.3f}, {null_reference['ci_high']:.3f}] over {null_reference['n']} passages",
            f"- null reference correlation dropped {validity['null_dropped']} of "
            f"{validity['null_attempted']} passages (constant accuracy vector, undefined Spearman)",
            "- both means are conditional on non-constant passages; dropped passages are excluded, not zero",
            f"- Cohen's kappa, best self-edit by generated vs gold questions: "
            f"{validity['best_self_edit_kappa']:.3f} over {validity['kappa_items']} self-edit labels",
        ]

    lines += ["", "## RQ1 primary criteria (PREREGISTRATION.md)", ""]
    for verdict in summary["rq1_verdicts"]:
        detail = f"({verdict['reason']})" if verdict["reason"] else ""
        if verdict["id"] == "RQ1-spearman" and verdict["ci_low"] is not None:
            detail = f"CI low {verdict['ci_low']:.3f}"
        if verdict["id"] == "RQ1-correctness" and verdict["mean_correctness"] is not None:
            detail = f"correctness {percent(verdict['mean_correctness'])}%"
        lines.append(f"- {verdict['id']}: {verdict['statement']} -> {verdict['verdict']} {detail}".rstrip())

    lines += ["", "## Outer loop training", ""]
    training_rows = summary["outer_training"]
    if not training_rows:
        lines.append("outer summaries not present")
    else:
        lines += [
            "| iteration | passages | reward % | tie % | mean margin | kept |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in training_rows:
            lines.append(
                f"| {row['iteration']} | {row['passages']} | {percent(row['reward_rate'])} | "
                f"{percent(row['tie_rate'])} | {row['mean_margin']:.3f} | {row['kept']} |"
            )

    lines += ["", "## Cost and compute", ""]
    cost = summary["cost"]
    grader = cost["grader"]
    if grader is None:
        lines.append("- grader usage not present")
    else:
        lines += [
            f"- grader calls: {grader['calls']}, input tokens: {grader['input_tokens']}, "
            f"output tokens: {grader['output_tokens']}, cost: {grader['cost_usd']:.2f} USD",
            f"- grader models: {', '.join(grader['models'])}",
        ]
    for stage_name, stage in (
        ("evaluation", cost["evaluation"]),
        ("SE-RL training", cost["se_rl"]),
        ("outer loop", cost["outer"]),
        ("total", cost["total"]),
    ):
        lines.append(f"- {stage_name} TTT: {stage['ttt_count']}, GPU hours: {stage['gpu_hours']:.2f}")
    if cost["pilot_ttt_seconds"] is None:
        lines.append("- pilot not present")
    else:
        lines.append(f"- pilot TTT seconds: {cost['pilot_ttt_seconds']:.1f}")

    lines += ["", "## Judge validation", ""]
    validation = summary["judge_validation"]
    if validation is None:
        lines.append("judge validation not present")
    else:
        lines.append(
            f"- cross model {validation['cross_model']} on n={validation['n']}: kappa {validation['kappa']:.3f}, "
            f"agreement {percent(validation['agreement'])}%, self-flip rate {percent(validation['self_flip_rate'])}%"
        )
    lines.append("")
    return "\n".join(lines)


def load_serl_rounds(run_dir):
    return [json.loads(path.read_text(encoding="utf-8")) for path in (run_dir / "serl").glob("*_round*.json")]


def stage_ttt_totals(stage_records):
    # older runs wrote these files before ttt_count/gpu_seconds were tracked
    return {
        "ttt_count": sum(record.get("ttt_count", 0) for record in stage_records),
        "gpu_hours": sum(record.get("gpu_seconds", 0) for record in stage_records) / 3600.0,
    }


def merged_grader_usage(run_dir):
    """The outer loop and the evaluation stage each keep their own tally."""
    parts = [read_optional(run_dir / name)
             for name in ("grader_usage.json", "grader_usage_outer.json")]
    parts = [part for part in parts if part]
    if not parts:
        return None
    return {
        "calls": sum(part["calls"] for part in parts),
        "input_tokens": sum(part["input_tokens"] for part in parts),
        "output_tokens": sum(part["output_tokens"] for part in parts),
        "cost_usd": round(sum(part["cost_usd"] for part in parts), 4),
        "models": sorted({model for part in parts for model in part["models"]}),
    }


def main():
    run_dir = paths.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    conditions = load_conditions(run_dir)
    accuracy_tables = {name: accuracy_by_passage(condition) for name, condition in conditions.items()}
    seed_table = seed_mean_accuracies(accuracy_tables)
    if seed_table:
        accuracy_tables[SEED_MEAN] = seed_table
    title_lookup = {}
    for condition in conditions.values():
        title_lookup.update(title_by_key(condition))
    outer_records = load_outer_records(run_dir)
    outer_summaries = load_outer_summaries(run_dir)
    pilot = read_optional(run_dir / "dev" / "pilot.json")
    proxy_validity_result = proxy_validity(outer_records)
    evaluation_cost = stage_ttt_totals(conditions.values())
    se_rl_cost = stage_ttt_totals(load_serl_rounds(run_dir))
    outer_cost = stage_ttt_totals(outer_summaries)
    total_cost = {
        "ttt_count": evaluation_cost["ttt_count"] + se_rl_cost["ttt_count"] + outer_cost["ttt_count"],
        "gpu_hours": evaluation_cost["gpu_hours"] + se_rl_cost["gpu_hours"] + outer_cost["gpu_hours"],
    }
    summary = {
        "run": {
            "name": config.RUN_NAME,
            "model": config.MODEL_NAME,
            "judge_model": config.GRADER_MODEL,
            "date": date.today().isoformat(),
            "note": HEADER_NOTE,
        },
        "conditions": condition_rows(conditions, accuracy_tables, title_lookup),
        "hypotheses": run_hypothesis_sequence(accuracy_tables, title_lookup),
        "generation_quality": generation_quality(outer_records),
        "proxy_validity": proxy_validity_result,
        "rq1_verdicts": rq1_verdicts(proxy_validity_result, outer_records),
        "outer_training": outer_summaries,
        "cost": {
            "grader": merged_grader_usage(run_dir),
            "evaluation": evaluation_cost,
            "se_rl": se_rl_cost,
            "outer": outer_cost,
            "total": total_cost,
            "pilot_ttt_seconds": None if pilot is None else pilot["ttt_seconds"],
        },
        "judge_validation": read_optional(run_dir / "judge" / "validation.json"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(f"wrote {run_dir / 'summary.json'} and {run_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
