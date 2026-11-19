"""Freeze an SQA3D validation subset disjoint from the inspected ScanQA split.

The selection is label-free. A question is eligible only when its SQA3D
validation scene is absent from the SQA3D train/test files and from the 71
ScanQA validation scenes used during GeoGate development.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


EXPECTED_QUESTIONS = 1_749
EXPECTED_SCENES = 35


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json_or_jsonl(path: Path) -> list[dict]:
    raw = Path(path).read_text(encoding="utf-8")
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError:
        rows = [
            json.loads(line)
            for line in raw.splitlines()
            if line.strip()
        ]
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or not all(
        isinstance(row, dict) for row in rows
    ):
        raise ValueError(f"Expected a JSON array or JSONL objects: {path}")
    return rows


def require_unique_question_ids(rows: list[dict], label: str) -> dict[str, dict]:
    keyed = {}
    for row in rows:
        question_id = str(row["question_id"])
        if question_id in keyed:
            raise ValueError(f"Duplicate question ID in {label}: {question_id}")
        keyed[question_id] = row
    return keyed


def scene_ids(rows: list[dict]) -> set[str]:
    scenes = {str(row["scene_id"]) for row in rows}
    if "" in scenes:
        raise ValueError("Empty scene ID")
    return scenes


def digest_strings(values: list[str]) -> str:
    payload = "".join(f"{value}\n" for value in values)
    return digest_bytes(payload.encode("utf-8"))


def build(
    *,
    sqa3d_val_question: Path,
    sqa3d_train: Path,
    sqa3d_test_question: Path,
    scanqa_val_question: Path,
) -> tuple[list[dict], dict]:
    val = load_json_or_jsonl(sqa3d_val_question)
    train = load_json_or_jsonl(sqa3d_train)
    test = load_json_or_jsonl(sqa3d_test_question)
    scanqa = load_json_or_jsonl(scanqa_val_question)

    val_by_id = require_unique_question_ids(val, "SQA3D validation")
    if len({str(row.get("id")) for row in train}) != len(train):
        raise ValueError("SQA3D train IDs are not unique")
    require_unique_question_ids(test, "SQA3D test")
    require_unique_question_ids(scanqa, "ScanQA validation")

    val_scenes = scene_ids(val)
    train_scenes = scene_ids(train)
    test_scenes = scene_ids(test)
    scanqa_scenes = scene_ids(scanqa)

    excluded = train_scenes | test_scenes | scanqa_scenes
    selected_scenes = sorted(val_scenes - excluded)
    selected = [
        row for row in val if str(row["scene_id"]) in set(selected_scenes)
    ]
    if len(selected_scenes) != EXPECTED_SCENES:
        raise ValueError(
            f"Expected {EXPECTED_SCENES} scenes, found {len(selected_scenes)}"
        )
    if len(selected) != EXPECTED_QUESTIONS:
        raise ValueError(
            f"Expected {EXPECTED_QUESTIONS} questions, found {len(selected)}"
        )
    if len({str(row["question_id"]) for row in selected}) != len(selected):
        raise ValueError("Selected SQA3D question IDs are not unique")

    question_ids = sorted(str(row["question_id"]) for row in selected)
    selected_scenes = sorted({str(row["scene_id"]) for row in selected})
    type_counts = Counter(int(row["type"]) for row in selected)
    manifest = {
        "schema": "geogate-sqa3d-scene-disjoint-manifest-v1",
        "scope": (
            "Label-free SQA3D validation selection. The frozen subset is "
            "disjoint from the SQA3D train/test files and from the 71 "
            "ScanQA validation scenes used during GeoGate development."
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
            )
        },
        "counts": {
            "sqa3d_val_questions": len(val),
            "sqa3d_val_scenes": len(val_scenes),
            "selected_questions": len(selected),
            "selected_scenes": len(selected_scenes),
            "sqa3d_train_scenes": len(train_scenes),
            "sqa3d_test_scenes": len(test_scenes),
            "scanqa_val_scenes": len(scanqa_scenes),
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
        },
    }
    return selected, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqa3d-val-question", type=Path, required=True)
    parser.add_argument("--sqa3d-train", type=Path, required=True)
    parser.add_argument("--sqa3d-test-question", type=Path, required=True)
    parser.add_argument("--scanqa-val-question", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.subset, args.manifest):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    selected, manifest = build(
        sqa3d_val_question=args.sqa3d_val_question.resolve(),
        sqa3d_train=args.sqa3d_train.resolve(),
        sqa3d_test_question=args.sqa3d_test_question.resolve(),
        scanqa_val_question=args.scanqa_val_question.resolve(),
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
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
