"""Strict audit and max-t family correction for the paired layer sweep."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from analysis.audit_selection_adjusted_inference import max_t_family
from analysis.layer_sweep_protocol import (
    DEFAULT_LAYERS,
    grouped_rows,
    parse_layers,
)
from analysis.paired_cider_bootstrap import cider_contributions, simple_tokenize
from tools.scanqa_score import clean_answer


ITERATIONS = 10_000
SEED = 20260920


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit_groups(
    grouped: dict,
    *,
    questions: dict[str, dict],
    ground_truth: dict[str, list[str]],
    layers: tuple[int, ...],
    keep_ratio: float,
) -> tuple[dict[str, np.ndarray], list[str]]:
    conditions = ("native",) + tuple(f"L{layer}" for layer in layers)
    question_ids = sorted(grouped)
    if set(question_ids) != set(questions) or set(question_ids) != set(ground_truth):
        raise ValueError("Question and ground-truth key sets differ")
    values = {condition: [] for condition in conditions}
    predictions_by_condition = {condition: {} for condition in conditions}
    scenes = []
    targets = {
        question_id: [
            simple_tokenize(clean_answer(answer))
            for answer in ground_truth[question_id]
        ]
        for question_id in question_ids
    }
    for question_id in question_ids:
        group = grouped[question_id]
        native = group["native"]
        full = int(native["full_tokens"])
        spatial = int(native["input_visual_tokens"])
        start = int(native["span_start"])
        text = int(native["text_tokens"])
        scene = str(questions[question_id]["video"])
        if native["scene_id"] != scene:
            raise ValueError(f"Scene mismatch: {question_id}")
        if full != spatial + text or start < 0 or start + spatial > full:
            raise ValueError(f"Invalid spatial span: {question_id}")
        if native["prune_layer"] is not None:
            raise ValueError("Native condition has a pruning layer")
        if native["selected_original_indices"] != list(range(full)):
            raise ValueError("Native selection is not the full sequence")
        for condition in conditions:
            row = group[condition]
            if (
                str(row["question_id"]) != question_id
                or row["scene_id"] != scene
                or row["prompt"] != native["prompt"]
                or row["prompt_sha256"] != native["prompt_sha256"]
                or row["seed"] != native["seed"]
                or row["input_tensor_hashes"] != native["input_tensor_hashes"]
                or row["expanded_input_sha256"] != native["expanded_input_sha256"]
                or int(row["input_visual_tokens"]) != spatial
                or int(row["span_start"]) != start
                or int(row["text_tokens"]) != text
                or int(row["full_tokens"]) != full
            ):
                raise ValueError(f"Unmatched shared input: {question_id}/{condition}")
            if condition == "native":
                layer = None
                keep = spatial
            else:
                layer = int(condition[1:])
                keep = max(1, int(round(keep_ratio * spatial)))
            reduced = full - spatial + keep
            expected_lengths = (
                [full] * 32
                if layer is None
                else [full] * (layer + 1) + [reduced] * (31 - layer)
            )
            indices = row["selected_original_indices"]
            if (
                row["prune_layer"] != layer
                or int(row["actual_kept_spatial_tokens"]) != keep
                or row["prefill_layer_input_lengths"] != expected_lengths
                or row["prefill_kv_lengths"] != [reduced] * 32
                or row["single_token_decode_verified"] is not True
                or indices != sorted(set(indices))
                or len(indices) != reduced
                or any(index < 0 or index >= full for index in indices)
                or sum(start <= index < start + spatial for index in indices) != keep
            ):
                raise ValueError(
                    f"Invalid trace or selection: {question_id}/{condition}"
                )
        for condition in conditions:
            answer = clean_answer(group[condition]["text"])
            references = [
                clean_answer(answer) for answer in ground_truth[question_id]
            ]
            predictions_by_condition[condition][question_id] = answer
            values[condition].append(int(answer in references))
        scenes.append(scene)
    if len(set(scenes)) != 71:
        raise ValueError("Expected 71 scene clusters")
    ciders = {}
    for condition in conditions:
        hypotheses = {
            question_id: [
                simple_tokenize(predictions_by_condition[condition][question_id])
            ]
            for question_id in question_ids
        }
        _, contributions = cider_contributions(targets, hypotheses)
        ciders[condition] = np.asarray(
            [float(contributions[question_id]) for question_id in question_ids],
            dtype=np.float64,
        )
    return (
        {
            condition: np.asarray(values[condition], dtype=np.float64)
            for condition in conditions
        },
        scenes,
        ciders,
    )


def build_audit(
    *,
    input_path: Path,
    questions_path: Path,
    answers_path: Path,
    layers: tuple[int, ...],
    keep_ratio: float,
    iterations: int,
    seed: int,
) -> dict:
    rows = load_rows(input_path)
    questions = {
        str(row["question_id"]): row
        for row in json.loads(questions_path.read_text(encoding="utf-8"))
    }
    ground_truth = {
        str(row["question_id"]): list(row["text"])
        for row in json.loads(answers_path.read_text(encoding="utf-8"))
    }
    conditions = ("native",) + tuple(f"L{layer}" for layer in layers)
    grouped = grouped_rows(rows, conditions)
    if len(grouped) != 4_675:
        raise ValueError(f"Expected 4,675 questions, got {len(grouped)}")
    correctness, scenes, cider = audit_groups(
        grouped,
        questions=questions,
        ground_truth=ground_truth,
        layers=layers,
        keep_ratio=keep_ratio,
    )
    scene_array = np.asarray(scenes)
    all_contrasts = {
        f"{left} - {right}": (left, right)
        for left, right in itertools.combinations(conditions, 2)
    }
    primary_contrasts = {
        f"L14 - {condition}": ("L14", condition)
        for condition in conditions
        if condition != "L14"
    }
    native_contrasts = {
        f"{condition} - native": (condition, "native")
        for condition in conditions
        if condition != "native"
    }
    pairwise = max_t_family(
        correctness,
        scene_array,
        all_contrasts,
        iterations=iterations,
        seed=seed,
    )
    primary = max_t_family(
        correctness,
        scene_array,
        primary_contrasts,
        iterations=iterations,
        seed=seed,
    )
    native = max_t_family(
        correctness,
        scene_array,
        native_contrasts,
        iterations=iterations,
        seed=seed,
    )
    primary_lower = [
        row["max_t_95ci_em_points"][0]
        for row in primary["contrasts"].values()
    ]
    if all(lower > 0.0 for lower in primary_lower):
        decision = "supports_familywise_l14_advantage"
    elif any(lower > 0.0 for lower in primary_lower):
        decision = "limited_familywise_l14_advantage"
    else:
        decision = "does_not_support_familywise_l14_advantage"
    return {
        "schema": "geogate-layer-sweep-paired-audit-v1",
        "state": "passed",
        "scope": (
            "Strict audit and scene-clustered studentized max-t correction "
            "over a shared-input paired rerun of the measured 11-layer curve. "
            "It is not an independent holdout or a complete 32-layer search."
        ),
        "counts": {
            "questions": len(grouped),
            "scenes": len(set(scenes)),
            "layers": len(layers),
            "conditions": len(conditions),
            "rows": len(rows),
        },
        "config": {
            "layers": list(layers),
            "keep_ratio": keep_ratio,
            "iterations": iterations,
            "bootstrap_seed": seed,
        },
        "arms": {
            condition: {
                "EM": float(correctness[condition].mean() * 100.0),
                "CIDEr": float(cider[condition].mean() * 100.0),
            }
            for condition in conditions
        },
        "native_contrasts": native,
        "primary_l14_contrasts": primary,
        "all_pairwise": pairwise,
        "decision": decision,
        "boundary": {
            "independent_confirmation": False,
            "complete_32_layer_search": False,
            "paper_updated": False,
            "earlier_reported_values_replaced": False,
        },
        "hashes": {
            "input": digest(input_path),
            "questions": digest(questions_path),
            "answers": digest(answers_path),
            "auditor": digest(Path(__file__)),
        },
    }


def render_report(result: dict) -> str:
    lines = [
        "# Paired Measured-Layer Sweep Audit",
        "",
        "Date: 2026-09-20 (Asia/Shanghai)",
        "",
        "This is a shared-input rerun of the measured 11-layer curve. It is "
        "not an independent holdout and does not replace the previously "
        "reported curve values.",
        "",
        "## Arms",
        "",
        "| Condition | EM | CIDEr |",
        "|---|---:|---:|",
    ]
    for condition, row in result["arms"].items():
        lines.append(f"| {condition} | {row['EM']:.4f} | {row['CIDEr']:.4f} |")
    lines.extend(
        [
            "",
            "## L14 Familywise Contrasts",
            "",
            "| Contrast | Delta EM | Max-t 95% CI | Decision |",
            "|---|---:|---:|---|",
        ]
    )
    for label, row in result["primary_l14_contrasts"]["contrasts"].items():
        interval = row["max_t_95ci_em_points"]
        lines.append(
            f"| {label} | {row['delta_em_points']:+.4f} | "
            f"[{interval[0]:+.4f}, {interval[1]:+.4f}] | "
            f"`{row['max_t_decision']}` |"
        )
    lines.extend(
        [
            "",
            f"Familywise decision: `{result['decision']}`.",
            "",
            "## Boundary",
            "",
            "The correction covers every pairwise contrast among the eleven "
            "measured layer candidates. It does not cover all 32 layers or "
            "the complete historical development search, and it remains an "
            "analysis on a previously inspected ScanQA split.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
    parser.add_argument("--keep-ratio", type=float, default=0.10)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    for path in (args.out, args.report):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    result = build_audit(
        input_path=args.input,
        questions_path=args.questions,
        answers_path=args.answers,
        layers=parse_layers(args.layers),
        keep_ratio=args.keep_ratio,
        iterations=args.iterations,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(render_report(result), encoding="utf-8")
    print(
        json.dumps(
            {"decision": result["decision"], "arms": result["arms"]},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
