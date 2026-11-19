#!/usr/bin/env python3
"""Prespecified answer-level analysis for the frozen v15 experiment."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis.audit_readout_generation_v15 import audit_rows
from analysis.readout_generation_v15_protocol import (
    BASELINE,
    FROZEN_SEED,
    LAYERS,
    RANDOM_SEEDS,
    RANKINGS,
    arm_by_name,
    primary_contrasts,
    top_over_bottom_damage,
)


def scene_interval(
    values,
    scenes,
    level,
    seed=FROZEN_SEED,
    iterations=10000,
):
    values = np.asarray(values, dtype=float)
    scenes = np.asarray(scenes)
    if (
        values.ndim != 1
        or len(values) != len(scenes)
        or not len(values)
        or not np.isfinite(values).all()
        or not 0 < level < 1
        or iterations < 1
    ):
        raise ValueError("Invalid v15 scene-bootstrap input")
    labels, inverse = np.unique(scenes, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=values)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(labels), size=(iterations, len(labels)))
    estimates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    alpha = (1.0 - level) / 2.0
    return {
        "mean": float(values.mean()),
        "level": level,
        "ci": np.quantile(
            estimates,
            [alpha, 1.0 - alpha],
        ).tolist(),
    }


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid v15 latency distribution")
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "iqr": float(
            np.quantile(values, 0.75)
            - np.quantile(values, 0.25)
        ),
    }


def load_gt(path):
    raw = path.read_text(encoding="utf-8").strip()
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError:
        rows = [
            json.loads(line)
            for line in raw.splitlines()
            if line.strip()
        ]
    if not isinstance(rows, list):
        raise ValueError("ScanQA ground truth must be a JSON array or JSONL")
    return rows


def summarize(
    rows,
    question_ids,
    gt_rows,
    cider_fn=None,
    clean_fn=None,
    tokenize_fn=None,
):
    if cider_fn is None or tokenize_fn is None:
        from analysis.paired_cider_bootstrap import (
            cider_contributions,
            simple_tokenize,
        )
        cider_fn = cider_fn or cider_contributions
        tokenize_fn = tokenize_fn or simple_tokenize
    if clean_fn is None:
        from tools.scanqa_score import clean_answer
        clean_fn = clean_answer
    audit = audit_rows(rows, question_ids, repeat_required=False)
    records = {
        (str(row["question_id"]), row["condition"]): row
        for row in rows
    }
    gt = {str(row["question_id"]): row["text"] for row in gt_rows}
    if len(gt) != len(gt_rows) or set(gt) != set(question_ids):
        raise ValueError("v15 ground-truth question set mismatch")
    scenes = [
        records[(str(question_id), BASELINE)]["scene_id"]
        for question_id in question_ids
    ]
    if len(set(scenes)) != 71:
        raise ValueError("Full v15 scoring requires 71 scenes")

    targets = {
        question_id: [
            tokenize_fn(clean_fn(text))
            for text in gt[question_id]
        ]
        for question_id in question_ids
    }
    all_targets = {}
    all_predictions = {}
    conditions = [BASELINE] + list(arm_by_name())
    for question_id in question_ids:
        for condition in conditions:
            key = f"{question_id}|{condition}"
            all_targets[key] = targets[question_id]
            prediction = records[(question_id, condition)]["answer"]
            all_predictions[key] = [
                tokenize_fn(clean_fn(prediction))
            ]
    _, cider_scores = cider_fn(all_targets, all_predictions)

    baseline = {"EM": [], "CIDEr": []}
    quality = {"EM": {}, "CIDEr": {}}
    for layer in LAYERS:
        for ranking in RANKINGS:
            seeds = RANDOM_SEEDS if ranking == "random" else (0,)
            for seed in seeds:
                quality["EM"][(layer, ranking, seed)] = []
                quality["CIDEr"][(layer, ranking, seed)] = []

    for question_id in question_ids:
        cleaned_refs = [clean_fn(text) for text in gt[question_id]]
        for condition in conditions:
            row = records[(question_id, condition)]
            em = 100.0 * int(clean_fn(row["answer"]) in cleaned_refs)
            cider = 100.0 * cider_scores[f"{question_id}|{condition}"]
            if condition == BASELINE:
                baseline["EM"].append(em)
                baseline["CIDEr"].append(cider)
            else:
                arm = arm_by_name()[condition]
                key = (arm["layer"], arm["ranking"], arm["seed"])
                quality["EM"][key].append(em)
                quality["CIDEr"][key].append(cider)

    for metric in quality:
        quality[metric] = {
            key: np.asarray(values, dtype=float)
            for key, values in quality[metric].items()
        }
        baseline[metric] = np.asarray(baseline[metric], dtype=float)

    result = {
        "state": "passed",
        "decision": None,
        "questions": len(question_ids),
        "scenes": len(set(scenes)),
        "rows": audit["rows"],
        "bootstrap_seed": FROZEN_SEED,
        "bootstrap_iterations": 10000,
        "primary_metric": "EM",
        "multiplicity": "two prespecified EM contrasts; 97.5% intervals",
        "scope": (
            "answer-level mechanism extension on inspected ScanQA validation; "
            "not an independent holdout"
        ),
        "baseline": {
            metric: float(values.mean())
            for metric, values in baseline.items()
        },
        "metrics": {},
        "latency_seconds": {},
    }
    for metric, values in quality.items():
        contrasts = primary_contrasts(values)
        result["metrics"][metric] = {
            "primary": {
                name: scene_interval(value, scenes, 0.975)
                for name, value in contrasts.items()
            },
            "top_over_random_damage": {
                f"L{layer}": scene_interval(
                    np.mean([
                        values[(layer, "random", seed)]
                        for seed in RANDOM_SEEDS
                    ], axis=0)
                    - values[(layer, "top", 0)],
                    scenes,
                    0.95,
                )
                for layer in LAYERS
            },
            "top_over_bottom_damage": {
                f"L{layer}": scene_interval(
                    top_over_bottom_damage(values, layer),
                    scenes,
                    0.95,
                )
                for layer in LAYERS
            },
            "arm_means": {
                f"L{layer}_{ranking}_s{seed}": float(
                    values[(layer, ranking, seed)].mean()
                )
                for layer in LAYERS
                for ranking in RANKINGS
                for seed in (
                    RANDOM_SEEDS if ranking == "random" else (0,)
                )
            },
        }

    for condition in conditions:
        result["latency_seconds"][condition] = distribution([
            records[(question_id, condition)]["latency_s"]
            for question_id in question_ids
        ])

    em_primary = result["metrics"]["EM"]["primary"]
    em_bottom = result["metrics"]["EM"][
        "top_over_bottom_damage"
    ]["L14"]
    cider_primary = result["metrics"]["CIDEr"]["primary"]
    cider_bottom = result["metrics"]["CIDEr"][
        "top_over_bottom_damage"
    ]["L14"]
    em_gate = (
        all(item["ci"][0] > 0 for item in em_primary.values())
        and em_bottom["ci"][0] > 0
    )
    cider_direction = (
        all(item["mean"] > 0 for item in cider_primary.values())
        and cider_bottom["mean"] > 0
    )
    result["promotion_gate"] = {
        "both_em_primary_lower_bounds_gt_zero": all(
            item["ci"][0] > 0 for item in em_primary.values()
        ),
        "l14_em_bottom_minus_top_lower_bound_gt_zero": (
            em_bottom["ci"][0] > 0
        ),
        "cidEr_direction_consistent": cider_direction,
    }
    result["decision"] = (
        "supports_answer_level_layer_specific_functional_readout"
        if em_gate and cider_direction else
        "does_not_support_prespecified_v15_answer_level_claim"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not manifest.get("is_full"):
        raise ValueError("Refusing to score non-full v15 output")
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = summarize(
        rows,
        [str(value) for value in manifest["question_ids"]],
        load_gt(args.gt),
    )
    result["hashes"] = {
        "input": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "manifest": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "gt": hashlib.sha256(args.gt.read_bytes()).hexdigest(),
        "scorer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
