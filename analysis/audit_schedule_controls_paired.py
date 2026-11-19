#!/usr/bin/env python3
"""Paired intervals for the implemented ScanQA schedule controls.

The audit reuses frozen LLaVA-3D predictions on the inspected ScanQA
validation split. It checks strict question and prompt pairing, then reports
question-level and scene-cluster bootstrap intervals for EM and CIDEr.
It is a post-hoc comparison of implemented schedule approximations, not an
official reproduction, an independent confirmation, or a leaderboard result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from collections import defaultdict
from typing import Iterable

import numpy as np
from pycocoevalcap.cider.cider import Cider

ROOT = Path(__file__).resolve().parents[1]


ITERATIONS = 10000
BASE_SEED = 20260918
EXPECTED_QUESTIONS = 4675
EXPECTED_SCENES = 71

DEFAULT_OUT = (
    ROOT / "results/raw/followup/schedule_controls_paired_20260918.json"
)
DEFAULT_REPORT = ROOT / "results/SCHEDULE_CONTROLS_PAIRED_20260918.md"

ARMS = {
    "geogate_l14_r10": ROOT / "results/raw/full_attn.jsonl",
    "fastv_l2_r50": ROOT / "results/raw/full_fastv.jsonl",
    "pyramiddrop_l8_16_24": ROOT / "results/raw/pyr_full_851625.jsonl",
    "adatoken_expdecay": ROOT / "results/raw/adatoken_full_expdecay.jsonl",
}
GROUND_TRUTH = (
    ROOT
    / "third_party/LLaVA-3D/playground/data/annotations/"
    "llava3d_scanqa_val_answer.json"
)

COMPARISONS = {
    "geogate_minus_fastv_l2_r50": {
        "control": "fastv_l2_r50",
        "control_label": "FastV-style L2/R=0.50",
        "control_retention": "50%",
        "seed_offset": 0,
    },
    "geogate_minus_pyramiddrop": {
        "control": "pyramiddrop_l8_16_24",
        "control_label": "PyramidDrop-style L8/16/24",
        "control_retention": "12.5%",
        "seed_offset": 1,
    },
    "geogate_minus_adatoken": {
        "control": "adatoken_expdecay",
        "control_label": "AdaToken-3D-style exponential decay",
        "control_retention": "~10%",
        "seed_offset": 2,
    },
}

EXPECTED_ROUNDED = {
    "geogate_l14_r10": {"em": 27.79, "cider": 94.33},
    "fastv_l2_r50": {"em": 26.55, "cider": 90.44},
    "pyramiddrop_l8_16_24": {"em": 27.25, "cider": 93.80},
    "adatoken_expdecay": {"em": 26.95, "cider": 92.86},
}


def clean_answer(text: str) -> str:
    """Match tools/scanqa_score.py's LEO-style answer normalization."""
    text = text.lower()
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


def scene_id(question_id: str) -> str:
    match = re.fullmatch(r"val-(scene\d+)-(\d+)", str(question_id))
    if match is None:
        raise ValueError(f"Unexpected ScanQA question ID: {question_id}")
    return f"scannet/{match.group(1)}_00"


def simple_tokenize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def cider_contributions(
    targets: dict[str, list[str]],
    predictions: dict[str, list[str]],
) -> tuple[float, dict[str, float]]:
    if set(targets) != set(predictions):
        raise ValueError("CIDEr targets and predictions require identical keys")
    keys = sorted(targets)
    ordered_targets = {key: targets[key] for key in keys}
    ordered_predictions = {key: predictions[key] for key in keys}
    aggregate, scores = Cider().compute_score(
        ordered_targets,
        ordered_predictions,
    )
    return float(aggregate), dict(zip(keys, map(float, scores)))


def scene_cluster_bootstrap(
    deltas: np.ndarray,
    scenes: list[str],
    iterations: int,
    seed: int,
) -> tuple[list[float], float, float]:
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.ndim != 1 or len(deltas) == 0:
        raise ValueError("Deltas must be a nonempty one-dimensional array")
    if len(scenes) != len(deltas):
        raise ValueError("Each delta must have one scene label")
    if not np.all(np.isfinite(deltas)):
        raise ValueError("Deltas must be finite")

    grouped_values: dict[str, list[float]] = defaultdict(list)
    for value, scene in zip(deltas.tolist(), scenes):
        grouped_values[str(scene)].append(value)
    scene_ids = sorted(grouped_values)
    if len(scene_ids) < 2:
        raise ValueError("At least two scene clusters are required")

    sums = np.asarray(
        [sum(grouped_values[scene]) for scene in scene_ids],
        dtype=np.float64,
    )
    counts = np.asarray(
        [len(grouped_values[scene]) for scene in scene_ids],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0,
        len(scene_ids),
        size=(iterations, len(scene_ids)),
    )
    estimates = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975]).tolist()

    observed = float(deltas.mean())
    centered_null = estimates - observed
    extreme = np.count_nonzero(np.abs(centered_null) >= abs(observed))
    p_value = float((extreme + 1) / (iterations + 1))
    return [float(lower), float(upper)], observed, p_value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def unique_by_question(rows: Iterable[dict], label: str) -> dict[str, dict]:
    keyed = {}
    for row in rows:
        question_id = str(row["question_id"])
        if question_id in keyed:
            raise ValueError(f"Duplicate question ID in {label}: {question_id}")
        keyed[question_id] = row
    return keyed


def validate_pairing(arms: dict[str, dict[str, dict]]) -> list[str]:
    reference_name = "geogate_l14_r10"
    reference = arms[reference_name]
    if len(reference) != EXPECTED_QUESTIONS:
        raise ValueError(
            f"Expected {EXPECTED_QUESTIONS} questions; got {len(reference)}"
        )
    question_ids = sorted(reference)
    for name, arm in arms.items():
        if set(arm) != set(question_ids):
            missing = sorted(set(question_ids) - set(arm))
            extra = sorted(set(arm) - set(question_ids))
            raise ValueError(
                f"{name} question set mismatch; missing={missing[:5]}, "
                f"extra={extra[:5]}"
            )
        for question_id in question_ids:
            if arm[question_id].get("prompt") != reference[question_id].get(
                "prompt"
            ):
                raise ValueError(
                    f"{name} prompt mismatch: {question_id}"
                )
    return question_ids


def load_ground_truth(path: Path) -> dict[str, list[str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    keyed = {}
    for row in rows:
        question_id = str(row["question_id"])
        if question_id in keyed:
            raise ValueError(f"Duplicate ground-truth ID: {question_id}")
        keyed[question_id] = [str(text) for text in row["text"]]
    return keyed


def em_values(
    rows: dict[str, dict],
    question_ids: list[str],
    ground_truth: dict[str, list[str]],
) -> np.ndarray:
    values = []
    for question_id in question_ids:
        prediction = clean_answer(str(rows[question_id]["text"]))
        references = [
            clean_answer(text) for text in ground_truth[question_id]
        ]
        values.append(float(prediction in references))
    return np.asarray(values, dtype=np.float64)


def cider_values(
    rows: dict[str, dict],
    question_ids: list[str],
    ground_truth: dict[str, list[str]],
) -> tuple[float, np.ndarray]:
    targets = {
        question_id: [
            simple_tokenize(text) for text in ground_truth[question_id]
        ]
        for question_id in question_ids
    }
    predictions = {
        question_id: [simple_tokenize(str(rows[question_id]["text"]))]
        for question_id in question_ids
    }
    aggregate, contributions = cider_contributions(targets, predictions)
    values = np.asarray(
        [contributions[question_id] for question_id in question_ids],
        dtype=np.float64,
    )
    if not np.allclose(np.mean(values), aggregate, atol=1e-12, rtol=1e-12):
        raise ValueError("Per-question CIDEr contributions do not average "
                         "to the corpus score")
    return float(aggregate) * 100.0, values * 100.0


def question_bootstrap(
    deltas: np.ndarray,
    iterations: int,
    seed: int,
    chunk_size: int = 1024,
) -> tuple[list[float], float, float]:
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.ndim != 1 or len(deltas) == 0:
        raise ValueError("Deltas must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(deltas)):
        raise ValueError("Deltas must be finite")

    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, chunk_size):
        stop = min(iterations, start + chunk_size)
        sampled = rng.integers(
            0,
            len(deltas),
            size=(stop - start, len(deltas)),
        )
        estimates[start:stop] = deltas[sampled].mean(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975]).tolist()

    observed = float(deltas.mean())
    centered_null = estimates - observed
    extreme = np.count_nonzero(np.abs(centered_null) >= abs(observed))
    p_value = float((extreme + 1) / (iterations + 1))
    return [float(lower), float(upper)], observed, p_value


def metric_summary(
    reference: np.ndarray,
    candidate: np.ndarray,
    scenes: list[str],
    unit: str,
    iterations: int,
    seed: int,
) -> dict:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("Paired metric arrays must be aligned vectors")
    deltas = candidate - reference
    question_ci, delta, question_p = question_bootstrap(
        deltas,
        iterations,
        seed,
    )
    scene_ci, scene_delta, scene_p = scene_cluster_bootstrap(
        deltas,
        scenes,
        iterations,
        seed + 1,
    )
    if abs(delta - scene_delta) > 1e-12:
        raise ValueError("Question and scene point effects disagree")
    return {
        "unit": unit,
        "control_mean": float(reference.mean()),
        "geogate_mean": float(candidate.mean()),
        "delta_geogate_minus_control": delta,
        "question_bootstrap_95ci": question_ci,
        "question_bootstrap_two_sided_p": question_p,
        "scene_cluster_bootstrap_95ci": scene_ci,
        "scene_cluster_bootstrap_two_sided_p": scene_p,
        "wins": int(np.count_nonzero(deltas > 0)),
        "ties": int(np.count_nonzero(deltas == 0)),
        "losses": int(np.count_nonzero(deltas < 0)),
    }


def classify_scene_interval(summary: dict) -> str:
    lower, upper = summary["scene_cluster_bootstrap_95ci"]
    if lower > 0:
        return "positive"
    if upper < 0:
        return "negative"
    return "unresolved"


def build_audit(
    *,
    output: Path = DEFAULT_OUT,
    iterations: int = ITERATIONS,
    base_seed: int = BASE_SEED,
) -> dict:
    if iterations < 1000:
        raise ValueError("At least 1,000 bootstrap iterations are required")
    source_paths = dict(ARMS)
    source_paths["ground_truth_scanqa"] = GROUND_TRUTH
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source files: {missing}")

    raw_arms = {name: load_jsonl(path) for name, path in ARMS.items()}
    arms = {
        name: unique_by_question(rows, name)
        for name, rows in raw_arms.items()
    }
    question_ids = validate_pairing(arms)
    ground_truth = load_ground_truth(GROUND_TRUTH)
    if set(ground_truth) != set(question_ids):
        raise ValueError("Ground truth question set does not match predictions")
    scenes = [scene_id(question_id) for question_id in question_ids]
    if len(set(scenes)) != EXPECTED_SCENES:
        raise ValueError(
            f"Expected {EXPECTED_SCENES} scenes; got {len(set(scenes))}"
        )

    arm_results = {}
    em_by_arm = {}
    cider_by_arm = {}
    for name in (
        "geogate_l14_r10",
        "fastv_l2_r50",
        "pyramiddrop_l8_16_24",
        "adatoken_expdecay",
    ):
        em = em_values(arms[name], question_ids, ground_truth)
        cider, cider_contributions_by_question = cider_values(
            arms[name],
            question_ids,
            ground_truth,
        )
        em_by_arm[name] = em
        cider_by_arm[name] = cider_contributions_by_question
        if round(float(em.mean() * 100.0), 2) != EXPECTED_ROUNDED[name]["em"]:
            raise ValueError(
                f"{name} EM mismatch: {em.mean() * 100.0} does not round "
                f"to {EXPECTED_ROUNDED[name]['em']}"
            )
        if round(cider, 2) != EXPECTED_ROUNDED[name]["cider"]:
            raise ValueError(
                f"{name} CIDEr mismatch: {cider} does not round to "
                f"{EXPECTED_ROUNDED[name]['cider']}"
            )
        arm_results[name] = {
            "path": str(ARMS[name].relative_to(ROOT)),
            "sha256": sha256(ARMS[name]),
            "model_id": sorted(
                {str(row.get("model_id", "")) for row in raw_arms[name]}
            ),
            "questions": len(arms[name]),
            "em_percent": float(em.mean() * 100.0),
            "cider_percent": cider,
        }

    comparisons = {}
    geogate_em = em_by_arm["geogate_l14_r10"]
    geogate_cider = cider_by_arm["geogate_l14_r10"]
    for name, spec in COMPARISONS.items():
        control = spec["control"]
        seed = base_seed + int(spec["seed_offset"])
        em_summary = metric_summary(
            em_by_arm[control] * 100.0,
            geogate_em * 100.0,
            scenes,
            "EM percentage points",
            iterations,
            seed,
        )
        cider_summary = metric_summary(
            cider_by_arm[control],
            geogate_cider,
            scenes,
            "CIDEr points",
            iterations,
            seed + 100,
        )
        comparisons[name] = {
            "control": control,
            "control_label": spec["control_label"],
            "control_retention": spec["control_retention"],
            "geogate_retention": "10%",
            "direction": "GeoGate minus implemented schedule control",
            "em": em_summary,
            "cider": cider_summary,
            "scene_interval_decision": {
                "em": classify_scene_interval(em_summary),
                "cider": classify_scene_interval(cider_summary),
            },
        }

    return {
        "status": "post_hoc_schedule_control_audit",
        "created_for": "review-risk audit of implemented schedule controls",
        "scope": {
            "benchmark": "LLaVA-3D ScanQA validation",
            "questions": len(question_ids),
            "scenes": len(set(scenes)),
            "iterations": iterations,
            "base_seed": base_seed,
            "paired_question_bootstrap": True,
            "paired_scene_cluster_bootstrap": True,
            "prompt_identity_checked": True,
            "model_rerun": False,
            "official_reproduction": False,
            "independent_confirmation": False,
            "selection_adjusted": False,
            "notes": (
                "The ScanQA validation split was used for profiling and "
                "configuration selection. PyramidDrop and AdaToken-3D arms "
                "are implemented schedule approximations, not official "
                "reproductions. These post-hoc intervals do not establish "
                "external generalization or a leaderboard ranking."
            ),
        },
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "size": path.stat().st_size,
            }
            for name, path in source_paths.items()
        },
        "arms": arm_results,
        "comparisons": comparisons,
        "generated_by": "analysis/audit_schedule_controls_paired.py",
        "output_hint": str(output.relative_to(ROOT)),
    }


def render_report(audit: dict) -> str:
    lines = [
        "# Paired Schedule-Control Audit",
        "",
        "Status: `post_hoc_schedule_control_audit`.",
        "",
        "This CPU-only audit reuses frozen LLaVA-3D ScanQA predictions on the "
        "inspected validation split. Every comparison checks identical "
        "question IDs and prompts, and reports `GeoGate - control` with "
        "10,000 paired question-level and scene-cluster bootstrap samples. "
        "It is not a model rerun, an official reproduction, or an "
        "independent confirmation.",
        "",
        "| Arm | Retained at drop | EM | CIDEr |",
        "|---|---:|---:|---:|",
    ]
    arm_order = (
        ("geogate_l14_r10", "10%"),
        ("fastv_l2_r50", "50%"),
        ("pyramiddrop_l8_16_24", "12.5%"),
        ("adatoken_expdecay", "~10%"),
    )
    for name, retention in arm_order:
        arm = audit["arms"][name]
        lines.append(
            f"| {name} | {retention} | {arm['em_percent']:.4f} "
            f"| {arm['cider_percent']:.4f} |"
        )

    for metric, label in (("em", "EM"), ("cider", "CIDEr")):
        lines.extend([
            "",
            f"## {label}",
            "",
            "| Control | GeoGate - control | Question 95% CI "
            "| Scene-cluster 95% CI | Decision |",
            "|---|---:|---|---|---|",
        ])
        for comparison in audit["comparisons"].values():
            summary = comparison[metric]
            question_low, question_high = summary[
                "question_bootstrap_95ci"
            ]
            scene_low, scene_high = summary[
                "scene_cluster_bootstrap_95ci"
            ]
            lines.append(
                f"| {comparison['control_label']} "
                f"| {summary['delta_geogate_minus_control']:+.4f} "
                f"| [{question_low:+.4f}, {question_high:+.4f}] "
                f"| [{scene_low:+.4f}, {scene_high:+.4f}] "
                f"| `{comparison['scene_interval_decision'][metric]}` |"
            )

    lines.extend([
        "",
        "An `unresolved` decision means the scene-clustered interval includes "
        "zero. It is not evidence of equivalence. A `positive` decision means "
        "the sampled interval is entirely above zero for the frozen compared "
        "arms; it is still limited to this inspected split and these "
        "implemented approximations.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    args = parser.parse_args()

    output = args.output.resolve()
    report = args.report.resolve()
    audit = build_audit(
        output=output,
        iterations=args.iterations,
        base_seed=args.base_seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    report.write_text(render_report(audit), encoding="utf-8")
    print(json.dumps({
        "status": audit["status"],
        "comparisons": {
            name: {
                metric: {
                    "delta": row[metric][
                        "delta_geogate_minus_control"
                    ],
                    "scene_ci": row[metric][
                        "scene_cluster_bootstrap_95ci"
                    ],
                    "decision": row["scene_interval_decision"][metric],
                }
                for metric in ("em", "cider")
            }
            for name, row in audit["comparisons"].items()
        },
        "output": str(output),
        "report": str(report),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
