"""Freeze the unseen-question remainder of SQA3D validation.

This fold complements the prior SQA3D scene-disjoint run. Its 30 scenes
overlap the ScanQA validation scenes used during development, so it is a
question-level replication rather than a new independent-scene confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from analysis.build_sqa3d_scene_disjoint_manifest import (
    digest_strings,
    load_json_or_jsonl,
    require_unique_question_ids,
    scene_ids,
)


EXPECTED_QUESTIONS = 1_512
EXPECTED_SCENES = 30


def build(
    *,
    sqa3d_val_question: Path,
    sqa3d_train: Path,
    sqa3d_test_question: Path,
    scanqa_val_question: Path,
    prior_subset: Path,
    expected_questions: int = EXPECTED_QUESTIONS,
    expected_scenes: int = EXPECTED_SCENES,
) -> tuple[list[dict], dict]:
    val = load_json_or_jsonl(sqa3d_val_question)
    train = load_json_or_jsonl(sqa3d_train)
    test = load_json_or_jsonl(sqa3d_test_question)
    scanqa = load_json_or_jsonl(scanqa_val_question)
    prior = load_json_or_jsonl(prior_subset)

    val_by_id = require_unique_question_ids(val, "SQA3D validation")
    require_unique_question_ids(prior, "Prior SQA3D fold")
    require_unique_question_ids(test, "SQA3D test")
    require_unique_question_ids(scanqa, "ScanQA validation")
    if len({str(row.get("id")) for row in train}) != len(train):
        raise ValueError("SQA3D train IDs are not unique")

    prior_ids = {str(row["question_id"]) for row in prior}
    missing_prior = prior_ids - set(val_by_id)
    if missing_prior:
        raise ValueError(
            f"Prior fold contains {len(missing_prior)} non-validation IDs"
        )

    selected = [
        row for row in val if str(row["question_id"]) not in prior_ids
    ]
    selected_ids = {str(row["question_id"]) for row in selected}
    if selected_ids & prior_ids:
        raise ValueError("Replication fold overlaps the prior fold")
    if selected_ids | prior_ids != set(val_by_id):
        raise ValueError("Prior and replication folds do not partition validation")
    if len(selected) != expected_questions:
        raise ValueError(
            f"Expected {expected_questions} questions, found {len(selected)}"
        )

    val_scenes = scene_ids(val)
    train_scenes = scene_ids(train)
    test_scenes = scene_ids(test)
    scanqa_scenes = scene_ids(scanqa)
    prior_scenes = scene_ids(prior)
    selected_scenes = sorted({str(row["scene_id"]) for row in selected})
    if len(selected_scenes) != expected_scenes:
        raise ValueError(
            f"Expected {expected_scenes} scenes, found {len(selected_scenes)}"
        )
    if set(selected_scenes) != val_scenes - prior_scenes:
        raise ValueError("Replication scenes are not the validation remainder")

    question_ids = sorted(selected_ids)
    type_counts = Counter(int(row["type"]) for row in selected)
    manifest = {
        "schema": "geogate-sqa3d-validation-replication-manifest-v1",
        "scope": (
            "Label-free remainder of SQA3D validation after the prior "
            "scene-disjoint fold. Questions are unseen in the prior run, but "
            "all selected scenes overlap the ScanQA validation scenes used "
            "during development. This is a question-level replication, not "
            "an independent-scene confirmation."
        ),
        "inputs": {
            path.name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in (
                sqa3d_val_question,
                sqa3d_train,
                sqa3d_test_question,
                scanqa_val_question,
                prior_subset,
            )
        },
        "counts": {
            "sqa3d_val_questions": len(val),
            "sqa3d_val_scenes": len(val_scenes),
            "prior_questions": len(prior_ids),
            "prior_scenes": len(prior_scenes),
            "selected_questions": len(selected),
            "selected_scenes": len(selected_scenes),
            "sqa3d_train_scenes": len(train_scenes),
            "sqa3d_test_scenes": len(test_scenes),
            "scanqa_val_scenes": len(scanqa_scenes),
            "selected_overlap_prior_fold": len(
                selected_ids & prior_ids
            ),
            "selected_overlap_sqa3d_train": len(
                set(selected_scenes) & train_scenes
            ),
            "selected_overlap_sqa3d_test": len(
                set(selected_scenes) & test_scenes
            ),
            "selected_overlap_scanqa_val": len(
                set(selected_scenes) & scanqa_scenes
            ),
            "question_type_counts": {
                str(key): value for key, value in sorted(type_counts.items())
            },
        },
        "hashes": {
            "question_ids_sha256": digest_strings(question_ids),
            "scene_ids_sha256": digest_strings(selected_scenes),
        },
        "protocol": {
            "model": "3D-LLaVA-7B-LoRA",
            "baseline": {"prune_layer": -1},
            "geogate": {
                "prune_layer": 14,
                "keep_ratio": 0.10,
                "scorer": "attn",
            },
            "prompt_template": (
                "<image>\\n {question} Answer the question using a single "
                "word or phrase."
            ),
            "decoding": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": 64,
                "seed": 0,
            },
            "primary_metric": "official SQA3D EM",
            "primary_estimand": (
                "paired GeoGate-minus-baseline EM on the selected questions"
            ),
            "noninferiority_margin_em_points": -1.0,
            "promotion_rule": (
                "Pass only when the paired scene-cluster bootstrap 97.5% "
                "lower bound for GeoGate-minus-baseline EM is greater than "
                "-1.0 EM points."
            ),
        },
        "boundary": {
            "predictions_inspected": False,
            "paper_updated": False,
            "claim_eligible_before_completion": False,
            "noninferiority_is_operational": True,
            "independent_scene_confirmation": False,
            "question_level_replication": True,
        },
    }
    return selected, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqa3d-val-question", type=Path, required=True)
    parser.add_argument("--sqa3d-train", type=Path, required=True)
    parser.add_argument("--sqa3d-test-question", type=Path, required=True)
    parser.add_argument("--scanqa-val-question", type=Path, required=True)
    parser.add_argument("--prior-subset", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exposure-eval", type=Path, default=None)
    args = parser.parse_args()
    output_paths = [args.subset, args.manifest]
    if args.exposure_eval is not None:
        output_paths.append(args.exposure_eval)
    for path in output_paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    selected, manifest = build(
        sqa3d_val_question=args.sqa3d_val_question.resolve(),
        sqa3d_train=args.sqa3d_train.resolve(),
        sqa3d_test_question=args.sqa3d_test_question.resolve(),
        scanqa_val_question=args.scanqa_val_question.resolve(),
        prior_subset=args.prior_subset.resolve(),
    )
    args.subset.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.subset.write_text(
        json.dumps(selected, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.exposure_eval is not None:
        exposure_rows = [
            {
                "question_id": str(row["question_id"]),
                "video": str(row["scene_id"]),
                "text": str(row["text"]),
            }
            for row in selected
        ]
        args.exposure_eval.parent.mkdir(parents=True, exist_ok=True)
        args.exposure_eval.write_text(
            json.dumps(exposure_rows, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
