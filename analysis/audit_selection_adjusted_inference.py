#!/usr/bin/env python3
"""Scene-clustered max-t intervals for the frozen layer and budget choices.

This is a limited family-wise correction over the complete pairwise
contrasts among each frozen three-candidate family. It is CPU-only and does
not re-run a model. It does not reconstruct the full historical search,
provide independent confirmation, or turn the inspected ScanQA split into a
confirmatory holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = (
    ROOT / "results/raw/followup/selection_adjusted_inference_20260919.json"
)
DEFAULT_REPORT = ROOT / "results/SELECTION_ADJUSTED_INFERENCE_20260919.md"

LAYER_INPUT = (
    ROOT / "results/raw/followup/review_v4_audit_20260910/full.jsonl"
)
BUDGET_INPUTS = {
    "R=0.05": ROOT / "results/raw/followup/scanqa_L14_attn_R005_full.jsonl",
    "R=0.10": ROOT / "results/raw/full_attn.jsonl",
    "R=0.20": ROOT / "results/raw/followup/scanqa_L14_attn_R020_full.jsonl",
}
GROUND_TRUTH = (
    ROOT
    / "third_party/LLaVA-3D/playground/data/annotations/"
    "llava3d_scanqa_val_answer.json"
)

EXPECTED_LAYER_EM = {
    "L2 cost-matched": 26.67379679144385,
    "L8 cost-matched": 26.86631016042781,
    "L14/128": 27.893048128342247,
}
EXPECTED_BUDGET_EM = {
    "R=0.05": 27.72192513368984,
    "R=0.10": 27.786096256684488,
    "R=0.20": 27.67914438502674,
}

LAYER_CANDIDATES = ("L2_cost", "L8_cost", "L14_128")
BUDGET_CANDIDATES = ("R=0.05", "R=0.10", "R=0.20")
LAYER_CONTRASTS = {
    "L14/128 - L2 cost-matched": ("L14_128", "L2_cost"),
    "L14/128 - L8 cost-matched": ("L14_128", "L8_cost"),
    "L8 cost-matched - L2 cost-matched": ("L8_cost", "L2_cost"),
}
BUDGET_CONTRASTS = {
    "R=0.10 - R=0.05": ("R=0.10", "R=0.05"),
    "R=0.10 - R=0.20": ("R=0.10", "R=0.20"),
    "R=0.20 - R=0.05": ("R=0.20", "R=0.05"),
}

ITERATIONS = 10_000
SEED = 20260919


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_answer(text: str) -> str:
    """Match the frozen ScanQA normalization used by the paper."""
    text = str(text).lower()
    text = re.sub("[ ]+$", "", text)
    text = re.sub("^[ ]+", "", text)
    text = re.sub(" {2,}", " ", text)
    text = re.sub(r"\.[ ]{2,}", ". ", text)
    text = re.sub(r"[^a-zA-Z0-9,'\s\-:]+", "", text)
    text = re.sub("ç", "c", text)
    text = re.sub("’", "'", text)
    for source, target in (
        ("leftf", "left"),
        ("let", "left"),
        ("tehre", "there"),
        ("rigth", "right"),
        ("rght", "right"),
        ("behine", "behind"),
        ("tv", "TV"),
        ("chai", "chair"),
        ("wasing", "washing"),
        ("waslked", "walked"),
        ("oclock", "o'clock"),
    ):
        text = re.sub(rf"\b{source}\b", target, text)
    text = re.sub(r"\bo'[ ]+clock\b", "o'clock", text)
    for digit, word in enumerate(
        (
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
        )
    ):
        text = re.sub(rf"\b{digit}\b", word, text)
    return re.sub(r"\bnone\b", "zero", text)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def scene_id(question_id: str) -> str:
    match = re.fullmatch(r"val-(scene\d+)-\d+", str(question_id))
    if match is None:
        raise ValueError(f"Unexpected ScanQA question ID: {question_id}")
    return f"scannet/{match.group(1)}_00"


def load_ground_truth(path: Path) -> dict[str, list[str]]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    ground_truth = {}
    for row in rows:
        question_id = str(row["question_id"])
        if question_id in ground_truth:
            raise ValueError(f"Duplicate ground-truth ID: {question_id}")
        ground_truth[question_id] = [
            clean_answer(text) for text in row["text"]
        ]
    return ground_truth


def keyed_rows(rows: list[dict], label: str) -> dict[str, dict]:
    keyed = {}
    for row in rows:
        question_id = str(row["question_id"])
        if question_id in keyed:
            raise ValueError(f"Duplicate question ID in {label}: {question_id}")
        keyed[question_id] = row
    return keyed


def correctness(
    rows: dict[str, dict],
    question_ids: list[str],
    ground_truth: dict[str, list[str]],
) -> np.ndarray:
    return np.asarray(
        [
            int(clean_answer(rows[question_id]["text"]) in ground_truth[question_id])
            for question_id in question_ids
        ],
        dtype=np.float64,
    )


def question_means(
    correctness_by_arm: dict[str, np.ndarray],
    draw: np.ndarray,
    scene_sums: dict[str, np.ndarray],
    scene_counts: np.ndarray,
) -> dict[str, float]:
    del correctness_by_arm
    denominator = float(scene_counts[draw].sum())
    return {
        name: float(values[draw].sum() / denominator)
        for name, values in scene_sums.items()
    }


def max_t_family(
    correctness_by_arm: dict[str, np.ndarray],
    scenes: np.ndarray,
    contrasts: dict[str, tuple[str, str]],
    *,
    iterations: int,
    seed: int,
) -> dict:
    scene_ids = sorted(set(scenes.tolist()))
    if len(scene_ids) < 2:
        raise ValueError("At least two scene clusters are required")
    indices = [np.where(scenes == scene)[0] for scene in scene_ids]
    counts = np.asarray([len(group) for group in indices], dtype=np.float64)
    sums = {
        name: np.asarray(
            [values[group].sum() for group in indices],
            dtype=np.float64,
        )
        for name, values in correctness_by_arm.items()
    }
    means = {
        name: float(values.mean())
        for name, values in correctness_by_arm.items()
    }
    observed = {
        label: means[candidate] - means[reference]
        for label, (candidate, reference) in contrasts.items()
    }

    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0,
        len(scene_ids),
        size=(iterations, len(scene_ids)),
    )
    bootstrap = {
        label: np.empty(iterations, dtype=np.float64)
        for label in contrasts
    }
    for row, draw in enumerate(sampled):
        draw_means = question_means(
            correctness_by_arm,
            draw,
            sums,
            counts,
        )
        for label, (candidate, reference) in contrasts.items():
            bootstrap[label][row] = (
                draw_means[candidate] - draw_means[reference]
            )

    standard_errors = {}
    studentized = {}
    for label in contrasts:
        residual = bootstrap[label] - observed[label]
        se = float(np.std(residual, ddof=1))
        if se <= 0:
            raise ValueError(f"Non-positive bootstrap SE for {label}")
        standard_errors[label] = se
        studentized[label] = residual / se

    max_deviation = np.max(
        np.stack([np.abs(studentized[label]) for label in contrasts]),
        axis=0,
    )
    critical = float(np.quantile(max_deviation, 0.95))
    observed_max = max(
        abs(observed[label]) / standard_errors[label]
        for label in contrasts
    )
    family_p = float(
        (np.count_nonzero(max_deviation >= observed_max) + 1)
        / (iterations + 1)
    )

    rows = {}
    for label in contrasts:
        delta = observed[label]
        se = standard_errors[label]
        unadjusted_p = float(
            (
                np.count_nonzero(
                    np.abs(studentized[label])
                    >= abs(delta) / se
                )
                + 1
            )
            / (iterations + 1)
        )
        rows[label] = {
            "candidate": contrasts[label][0],
            "reference": contrasts[label][1],
            "delta_em_points": delta * 100.0,
            "scene_bootstrap_se_em_points": se * 100.0,
            "unadjusted_normal_95ci_em_points": [
                (delta - 1.96 * se) * 100.0,
                (delta + 1.96 * se) * 100.0,
            ],
            "unadjusted_two_sided_p": unadjusted_p,
            "max_t_95ci_em_points": [
                (delta - critical * se) * 100.0,
                (delta + critical * se) * 100.0,
            ],
            "max_t_decision": (
                "positive"
                if delta - critical * se > 0
                else "negative"
                if delta + critical * se < 0
                else "unresolved"
            ),
        }

    return {
        "questions": int(len(scenes)),
        "scenes": len(scene_ids),
        "iterations": iterations,
        "seed": seed,
        "arm_em_percent": {
            name: value * 100.0
            for name, value in sorted(means.items())
        },
        "max_t_critical_95": critical,
        "familywise_p": family_p,
        "contrasts": rows,
    }


def build_audit(
    *,
    iterations: int = ITERATIONS,
    seed: int = SEED,
) -> dict:
    if iterations < 1_000:
        raise ValueError("At least 1,000 bootstrap iterations are required")

    ground_truth = load_ground_truth(GROUND_TRUTH)
    layer_rows = load_jsonl(LAYER_INPUT)
    layer_records: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in layer_rows:
        question_id = str(row["question_id"])
        condition = str(row["condition"])
        if condition in LAYER_CANDIDATES:
            if condition in layer_records[question_id]:
                raise ValueError(
                    f"Duplicate layer row: {question_id} / {condition}"
                )
            layer_records[question_id][condition] = row

    layer_question_ids = sorted(layer_records)
    if len(layer_question_ids) != 4_675:
        raise ValueError("Expected 4,675 layer-selection questions")
    for question_id in layer_question_ids:
        if set(layer_records[question_id]) != set(LAYER_CANDIDATES):
            raise ValueError(f"Layer condition mismatch: {question_id}")
        reference = layer_records[question_id]["L2_cost"]
        for condition in LAYER_CANDIDATES:
            candidate = layer_records[question_id][condition]
            for field in ("prompt", "seed", "scene_id"):
                if candidate[field] != reference[field]:
                    raise ValueError(
                        f"Layer pairing mismatch for {question_id}: {field}"
                    )
            if (
                candidate["input_tensor_hashes"]
                != reference["input_tensor_hashes"]
            ):
                raise ValueError(
                    f"Layer input-tensor mismatch: {question_id}"
                )
            if (
                candidate["expanded_input_sha256"]
                != reference["expanded_input_sha256"]
            ):
                raise ValueError(
                    f"Layer expanded-input mismatch: {question_id}"
                )

    layer_correct = {
        condition: correctness(
            {
                question_id: layer_records[question_id][condition]
                for question_id in layer_question_ids
            },
            layer_question_ids,
            ground_truth,
        )
        for condition in LAYER_CANDIDATES
    }
    layer_scenes = np.asarray(
        [
            str(layer_records[question_id]["L2_cost"]["scene_id"])
            for question_id in layer_question_ids
        ]
    )
    if len(set(layer_scenes.tolist())) != 71:
        raise ValueError("Expected 71 layer-selection scenes")
    layer_observed = {
        condition: float(values.mean() * 100.0)
        for condition, values in layer_correct.items()
    }
    for condition, expected in EXPECTED_LAYER_EM.items():
        key = {
            "L2 cost-matched": "L2_cost",
            "L8 cost-matched": "L8_cost",
            "L14/128": "L14_128",
        }[condition]
        if abs(layer_observed[key] - expected) > 1e-8:
            raise ValueError(
                f"Layer EM mismatch for {condition}: "
                f"{layer_observed[key]} != {expected}"
            )

    budget_records = {}
    budget_question_ids = None
    for label, path in BUDGET_INPUTS.items():
        keyed = keyed_rows(load_jsonl(path), label)
        if budget_question_ids is None:
            budget_question_ids = sorted(keyed)
        elif sorted(keyed) != budget_question_ids:
            raise ValueError(f"Budget question set mismatch: {label}")
        budget_records[label] = keyed
    if budget_question_ids is None or len(budget_question_ids) != 4_675:
        raise ValueError("Expected 4,675 budget-selection questions")
    reference_prompts = {
        question_id: budget_records["R=0.10"][question_id]["prompt"]
        for question_id in budget_question_ids
    }
    for label, records in budget_records.items():
        for question_id in budget_question_ids:
            if records[question_id]["prompt"] != reference_prompts[question_id]:
                raise ValueError(
                    f"Budget prompt mismatch for {label}: {question_id}"
                )

    budget_correct = {
        label: correctness(
            budget_records[label],
            budget_question_ids,
            ground_truth,
        )
        for label in BUDGET_CANDIDATES
    }
    budget_scenes = np.asarray(
        [scene_id(question_id) for question_id in budget_question_ids]
    )
    if len(set(budget_scenes.tolist())) != 71:
        raise ValueError("Expected 71 budget-selection scenes")
    budget_observed = {
        label: float(values.mean() * 100.0)
        for label, values in budget_correct.items()
    }
    for label, expected in EXPECTED_BUDGET_EM.items():
        if abs(budget_observed[label] - expected) > 1e-8:
            raise ValueError(
                f"Budget EM mismatch for {label}: "
                f"{budget_observed[label]} != {expected}"
            )

    source_paths = {
        str(LAYER_INPUT.relative_to(ROOT)): LAYER_INPUT,
        **{
            str(path.relative_to(ROOT)): path
            for path in BUDGET_INPUTS.values()
        },
        str(GROUND_TRUTH.relative_to(ROOT)): GROUND_TRUTH,
    }
    return {
        "schema": "geogate-selection-adjusted-inference-v1",
        "state": "passed",
        "scope": (
            "CPU-only scene-clustered studentized max-t intervals for two "
            "limited candidate families on the inspected ScanQA validation "
            "split. Each family includes all pairwise contrasts among its "
            "three frozen candidates, but the correction is not the full "
            "historical search. It is not an independent holdout, external "
            "validation, or new model run."
        ),
        "sources": {
            path: {
                "bytes": source.stat().st_size,
                "sha256": digest_file(source),
            }
            for path, source in sorted(source_paths.items())
        },
        "layer_family": max_t_family(
            layer_correct,
            layer_scenes,
            LAYER_CONTRASTS,
            iterations=iterations,
            seed=seed,
        ),
        "budget_family": max_t_family(
            budget_correct,
            budget_scenes,
            BUDGET_CONTRASTS,
            iterations=iterations,
            seed=seed + 1,
        ),
        "boundary": {
            "limited_familywise_adjustment": True,
            "full_historical_selection_adjusted": False,
            "independent_confirmation": False,
            "paper_updated": False,
        },
    }


def render_report(audit: dict) -> str:
    lines = [
        "# Limited Selection-Adjusted Inference",
        "",
        "Date: 2026-09-19 (Asia/Shanghai)",
        "",
        "This CPU-only audit reuses the frozen ScanQA predictions. It applies "
        "a scene-clustered studentized max-$t$ correction to every pairwise "
        "contrast within each frozen three-candidate family.",
        "",
        "## Layer Selection",
        "",
        "| Contrast | EM points | Unadjusted 95% CI | Max-$t$ 95% CI | Decision |",
        "|---|---:|---:|---:|---|",
    ]
    for label, row in audit["layer_family"]["contrasts"].items():
        lines.append(
            f"| {label} | {row['delta_em_points']:+.4f} | "
            f"[{row['unadjusted_normal_95ci_em_points'][0]:.4f}, "
            f"{row['unadjusted_normal_95ci_em_points'][1]:.4f}] | "
            f"[{row['max_t_95ci_em_points'][0]:.4f}, "
            f"{row['max_t_95ci_em_points'][1]:.4f}] | "
            f"`{row['max_t_decision']}` |"
        )
    lines.extend([
        "",
        f"Familywise max-$t$ p-value: "
        f"`{audit['layer_family']['familywise_p']:.6f}`; critical value "
        f"`{audit['layer_family']['max_t_critical_95']:.6f}`.",
        "",
        "## Budget Selection",
        "",
        "| Contrast | EM points | Unadjusted 95% CI | Max-$t$ 95% CI | Decision |",
        "|---|---:|---:|---:|---|",
    ])
    for label, row in audit["budget_family"]["contrasts"].items():
        lines.append(
            f"| {label} | {row['delta_em_points']:+.4f} | "
            f"[{row['unadjusted_normal_95ci_em_points'][0]:.4f}, "
            f"{row['unadjusted_normal_95ci_em_points'][1]:.4f}] | "
            f"[{row['max_t_95ci_em_points'][0]:.4f}, "
            f"{row['max_t_95ci_em_points'][1]:.4f}] | "
            f"`{row['max_t_decision']}` |"
        )
    lines.extend([
        "",
        f"Familywise max-$t$ p-value: "
        f"`{audit['budget_family']['familywise_p']:.6f}`; critical value "
        f"`{audit['budget_family']['max_t_critical_95']:.6f}`.",
        "",
        "## Boundary",
        "",
        "The two central layer contrasts remain positive after this complete "
        "pairwise "
        "correction. The budget family remains unresolved: L14/R=0.10 does "
        "not significantly outperform its neighboring budgets, so it is a "
        "representative point in a quality plateau rather than a uniquely "
        "optimal budget.",
        "",
        "The correction covers all pairwise contrasts among the three frozen "
        "candidates in each family, but not every layer, scorer, budget, "
        "prompt, or threshold inspected while developing the method. It "
        "therefore does not provide complete selection-adjusted inference, "
        "an independent holdout, external confirmation, or evidence against "
        "unseen configurations.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    audit = build_audit(iterations=args.iterations, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(render_report(audit), encoding="utf-8")
    print(json.dumps({
        "state": audit["state"],
        "layer_decisions": {
            key: row["max_t_decision"]
            for key, row in audit["layer_family"]["contrasts"].items()
        },
        "budget_decisions": {
            key: row["max_t_decision"]
            for key, row in audit["budget_family"]["contrasts"].items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
