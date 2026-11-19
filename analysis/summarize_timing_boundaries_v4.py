"""Summarize per-session timing and quality without pooling questions across runs."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from analysis.paired_cider_bootstrap import cider_contributions, simple_tokenize
from tools.scanqa_score import clean_answer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def percentile(values, q):
    return float(np.quantile(np.asarray(values, dtype=float), q))


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": percentile(values, 0.95),
        "iqr": percentile(values, 0.75) - percentile(values, 0.25),
    }


def bootstrap_median(values, scenes=None, iterations=10000, seed=20260912):
    rng = random.Random(seed)
    values = np.asarray(values, dtype=float)
    if scenes is None:
        estimates = []
        for _ in range(iterations):
            sample = np.asarray([
                values[rng.randrange(len(values))] for _ in range(len(values))
            ])
            estimates.append(np.median(sample))
    else:
        grouped = defaultdict(list)
        for value, scene in zip(values, scenes):
            grouped[str(scene)].append(float(value))
        scene_ids = sorted(grouped)
        estimates = []
        for _ in range(iterations):
            sampled = [scene_ids[rng.randrange(len(scene_ids))] for _ in scene_ids]
            sample = [value for scene in sampled for value in grouped[scene]]
            estimates.append(np.median(sample))
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def bootstrap_mean_delta(deltas, scenes, iterations, seed):
    keys = sorted(deltas)
    values = np.asarray([deltas[key] for key in keys], dtype=float)
    rng = random.Random(seed)
    question_estimates = []
    for _ in range(iterations):
        sample = np.asarray([
            values[rng.randrange(len(values))] for _ in range(len(values))
        ])
        question_estimates.append(float(sample.mean()))
    grouped = defaultdict(list)
    for key in keys:
        grouped[str(scenes[key])].append(float(deltas[key]))
    scene_ids = sorted(grouped)
    scene_estimates = []
    for _ in range(iterations):
        sampled = [scene_ids[rng.randrange(len(scene_ids))] for _ in scene_ids]
        sample = [value for scene in sampled for value in grouped[scene]]
        scene_estimates.append(float(np.mean(sample)))
    return {
        "mean": float(values.mean()),
        "question_ci95": [percentile(question_estimates, 0.025),
                          percentile(question_estimates, 0.975)],
        "scene_ci95": [percentile(scene_estimates, 0.025),
                       percentile(scene_estimates, 0.975)],
    }


def load_json_or_jsonl(path):
    raw = Path(path).read_text(encoding="utf-8").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return parsed
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def summarize_run(path, gt_path, iterations, seed):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    rows = [row for row in rows if not row["warmup"]]
    grouped = defaultdict(dict)
    for row in rows:
        key = (str(row["question_id"]), row["condition"])
        if key in grouped:
            raise ValueError(f"Duplicate row: {key}")
        grouped[key] = row
    question_ids = sorted({qid for qid, _ in grouped})
    expected = {(qid, condition) for qid in question_ids
                for condition in ("baseline", "geogate")}
    if set(grouped) != expected:
        raise ValueError("Incomplete baseline/geogate pairs")

    gt = {str(row["question_id"]): row for row in load_json_or_jsonl(gt_path)}
    if set(gt) != set(question_ids):
        raise ValueError("Ground-truth question set mismatch")
    scenes = {qid: grouped[(qid, "baseline")]["scene_id"] for qid in question_ids}
    targets = {
        qid: [simple_tokenize(clean_answer(text)) for text in gt[qid]["text"]]
        for qid in question_ids
    }

    result = {
        "path": str(Path(path).resolve()),
        "sha256": digest(path),
        "n_questions": len(question_ids),
        "n_scenes": len(set(scenes.values())),
        "conditions": {},
        "paired": {},
        "quality": {},
    }
    metric_rows = {
        "generation_ms": lambda row: row["generation_s"] * 1000,
        "application_ms": lambda row: row["application_s"] * 1000,
        "preprocess_ms": lambda row: row["preprocess_s"] * 1000,
        "h2d_ms": lambda row: row["h2d_s"] * 1000,
        "decode_ms": lambda row: row["decode_s"] * 1000,
        "peak_allocated_absolute_mib": lambda row: row["peak_allocated_absolute_mib"],
        "peak_allocated_incremental_mib": lambda row: row["peak_allocated_incremental_mib"],
        "peak_reserved_absolute_mib": lambda row: row["peak_reserved_absolute_mib"],
        "peak_reserved_incremental_mib": lambda row: row["peak_reserved_incremental_mib"],
    }
    for condition in ("baseline", "geogate"):
        condition_rows = [grouped[(qid, condition)] for qid in question_ids]
        result["conditions"][condition] = {
            name: distribution([fn(row) for row in condition_rows])
            for name, fn in metric_rows.items()
        }

    for metric, fn in (
        ("generation_ms", metric_rows["generation_ms"]),
        ("application_ms", metric_rows["application_ms"]),
    ):
        ratios = [
            fn(grouped[(qid, "geogate")]) / fn(grouped[(qid, "baseline")])
            for qid in question_ids
        ]
        result["paired"][metric] = {
            "ratio_median": float(np.median(ratios)),
            "ratio_question_ci95": bootstrap_median(
                ratios, scenes=None, iterations=iterations, seed=seed
            ),
            "ratio_scene_ci95": bootstrap_median(
                [ratios[i] for i in range(len(question_ids))],
                scenes=[scenes[qid] for qid in question_ids],
                iterations=iterations,
                seed=seed + 1,
            ),
            "speedup_median": float(1.0 / np.median(ratios)),
        }

    for condition in ("baseline", "geogate"):
        predictions = {
            qid: clean_answer(grouped[(qid, condition)]["text"])
            for qid in question_ids
        }
        em = {
            qid: 100 * int(predictions[qid] in
                           [clean_answer(text) for text in gt[qid]["text"]])
            for qid in question_ids
        }
        cider_mean, cider = cider_contributions(
            targets,
            {qid: [simple_tokenize(predictions[qid])] for qid in question_ids},
        )
        result["quality"][condition] = {
            "EM": float(np.mean(list(em.values()))),
            "CIDEr": float(cider_mean * 100),
            "per_question_EM": em,
            "per_question_CIDEr": {qid: value * 100 for qid, value in cider.items()},
        }

    em_delta = {
        qid: result["quality"]["geogate"]["per_question_EM"][qid]
        - result["quality"]["baseline"]["per_question_EM"][qid]
        for qid in question_ids
    }
    cider_delta = {
        qid: result["quality"]["geogate"]["per_question_CIDEr"][qid]
        - result["quality"]["baseline"]["per_question_CIDEr"][qid]
        for qid in question_ids
    }
    result["quality"]["delta_EM"] = bootstrap_mean_delta(
        em_delta, scenes, iterations, seed + 2
    )
    result["quality"]["delta_CIDEr"] = bootstrap_mean_delta(
        cider_delta, scenes, iterations, seed + 3
    )
    for condition in ("baseline", "geogate"):
        result["quality"][condition].pop("per_question_EM")
        result["quality"][condition].pop("per_question_CIDEr")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    runs = [
        summarize_run(path, args.gt, args.iterations, args.seed + index * 100)
        for index, path in enumerate(args.input)
    ]
    across = {}
    for metric in ("generation_ms", "application_ms"):
        ratios = [run["paired"][metric]["ratio_median"] for run in runs]
        across[metric] = {
            "run_ratio_medians": ratios,
            "median_across_runs": float(np.median(ratios)),
            "min_across_runs": float(min(ratios)),
            "max_across_runs": float(max(ratios)),
            "all_lower_than_one": bool(all(value < 1 for value in ratios)),
        }
    for metric in ("EM", "CIDEr"):
        deltas = [run["quality"][f"delta_{metric}"]["mean"] for run in runs]
        across[f"delta_{metric}"] = {
            "run_means": deltas,
            "median_across_runs": float(np.median(deltas)),
            "min_across_runs": float(min(deltas)),
            "max_across_runs": float(max(deltas)),
        }

    payload = {
        "protocol": {
            "n_runs": len(runs),
            "pooling": "none; each run is reported separately",
            "warmup_removed": True,
            "paired_condition": "baseline versus geogate on identical questions",
        },
        "runs": runs,
        "across_runs": across,
        "hashes": {"gt": digest(args.gt)},
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
