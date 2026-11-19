"""Prespecified scene-bootstrap analysis for the frozen v13 probe."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis.audit_readout_logit_v13 import audit_rows
from analysis.readout_logit_v13_protocol import (
    LAYERS,
    RANDOM_SEEDS,
    RANKINGS,
    arm_by_name,
    primary_contrasts,
)


def scene_interval(values, scenes, level, seed=20260913, iterations=10000):
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
        raise ValueError("Invalid scene-bootstrap input")
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
        "ci": np.quantile(estimates, [alpha, 1.0 - alpha]).tolist(),
    }


def metric_values(records, question_ids, field):
    arms = arm_by_name()
    output = {}
    for arm in arms.values():
        key = (arm["layer"], arm["ranking"], arm["seed"])
        output[key] = np.asarray([
            float(records[(question_id, arm["name"])][field])
            for question_id in question_ids
        ])
    return output


def summarize(rows, question_ids):
    audit = audit_rows(rows, question_ids)
    records = {
        (str(row["question_id"]), row["condition"]): row
        for row in rows
    }
    scenes = [records[(str(question_id), "baseline")]["scene_id"]
              for question_id in question_ids]
    if len(set(scenes)) != 71:
        raise ValueError("Full v13 scoring requires 71 scenes")

    metrics = {
        "logprob_damage": {
            key: -values
            for key, values in metric_values(
                records, question_ids, "intervention_token_logprob"
            ).items()
        },
        "margin_damage": {
            key: -values
            for key, values in metric_values(
                records, question_ids, "intervention_token_margin"
            ).items()
        },
        "js_divergence": metric_values(
            records, question_ids, "js_divergence"
        ),
        "hidden_cosine_displacement": {
            key: 1.0 - values
            for key, values in metric_values(
                records, question_ids, "final_prompt_hidden_cosine"
            ).items()
        },
        "argmax_flip": {
            key: values.astype(float)
            for key, values in metric_values(
                records, question_ids, "argmax_flipped"
            ).items()
        },
    }
    baseline_logprob = np.asarray([
        float(records[(str(question_id), "baseline")]["baseline_token_logprob"])
        for question_id in question_ids
    ])
    baseline_margin = np.asarray([
        float(records[(str(question_id), "baseline")]["baseline_token_margin"])
        for question_id in question_ids
    ])
    metrics["logprob_damage"] = {
        key: baseline_logprob + values
        for key, values in metrics["logprob_damage"].items()
    }
    metrics["margin_damage"] = {
        key: baseline_margin + values
        for key, values in metrics["margin_damage"].items()
    }

    result = {
        "state": "passed",
        "decision": None,
        "questions": len(question_ids),
        "scenes": len(set(scenes)),
        "rows": audit["rows"],
        "bootstrap_seed": 20260913,
        "bootstrap_iterations": 10000,
        "primary_metric": "logprob_damage",
        "multiplicity": "two prespecified contrasts; 97.5% intervals",
        "metrics": {},
    }
    for metric, values in metrics.items():
        contrasts = primary_contrasts(values)
        result["metrics"][metric] = {
            "primary": {
                name: scene_interval(value, scenes, 0.975)
                for name, value in contrasts.items()
            },
            "top_over_random": {
                f"L{layer}": scene_interval(
                    values[(layer, "top", 0)]
                    - np.mean([
                        values[(layer, "random", seed)]
                        for seed in RANDOM_SEEDS
                    ], axis=0),
                    scenes,
                    0.95,
                )
                for layer in LAYERS
            },
            "top_over_bottom": {
                f"L{layer}": scene_interval(
                    values[(layer, "top", 0)]
                    - values[(layer, "bottom", 0)],
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
    primary = result["metrics"]["logprob_damage"]["primary"]
    top_bottom_l14 = result["metrics"]["logprob_damage"][
        "top_over_bottom"
    ]["L14"]
    both = all(item["ci"][0] > 0 for item in primary.values())
    only_l20 = (
        primary["L14_minus_L20"]["ci"][0] > 0
        and primary["L14_minus_L8"]["ci"][0] <= 0
    )
    direction = top_bottom_l14["mean"] > 0
    if both and direction:
        result["decision"] = "supports_layer_specific_functional_readout"
    elif only_l20 and direction:
        result["decision"] = "supports_late_regime_not_unique_onset"
    else:
        result["decision"] = "does_not_support_prespecified_v13_claim"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    question_ids = [str(value) for value in manifest["question_ids"]]
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = summarize(rows, question_ids)
    result["hashes"] = {
        "input": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "manifest": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "scorer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
