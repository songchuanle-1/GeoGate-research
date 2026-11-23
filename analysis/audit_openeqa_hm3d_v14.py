#!/usr/bin/env python3
"""Structural audit for paired OpenEQA-HM3D v14 predictions."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


FROZEN_DATASET_SHA256 = (
    "afbcc703c50c121451a912181a4ae91f4ad94866b6dc48bdd7e9f566688fc8c9"
)
FROZEN_SEED = 20260913
FROZEN_QUESTIONS = 557
FROZEN_EPISODES = 63
CONDITIONS = ("native", "geogate_L14_R010")
FORWARD_PATHS = {
    "native": "native_unpatched_llama_forward",
    "geogate_L14_R010": "geogate_patched_llama_forward",
}


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(
                    f"Line {line_number} is not a JSON object"
                )
            rows.append(row)
    return rows


def shared_projection(row):
    return {
        "episode": row.get("episode"),
        "scene_id": row.get("scene_id"),
        "category": row.get("category"),
        "question": row.get("question"),
        "prompt": row.get("prompt"),
        "frame_selection": row.get("frame_selection"),
        "input_tensor_sha256": row.get("input_tensor_sha256"),
        "expanded_input_sha256": row.get("expanded_input_sha256"),
        "full_tokens": row.get("full_tokens"),
        "spatial_start": row.get("spatial_start"),
        "input_visual_tokens": row.get("input_visual_tokens"),
        "seed": row.get("seed"),
        "deterministic_scene_cap": row.get(
            "deterministic_scene_cap"
        ),
    }


def audit(raw_path, manifest_path, require_full=False):
    errors = []
    manifest = json.loads(
        Path(manifest_path).read_text(encoding="utf-8")
    )
    rows = read_jsonl(raw_path)

    if manifest.get("dataset", {}).get("sha256") != (
        FROZEN_DATASET_SHA256
    ):
        errors.append("manifest_dataset_sha256_mismatch")
    protocol = manifest.get("protocol", {})
    if protocol.get("seed") != FROZEN_SEED:
        errors.append("manifest_seed_mismatch")
    if protocol.get("conditions") != list(CONDITIONS):
        errors.append("manifest_conditions_mismatch")
    if protocol.get("condition_order") != "native_then_geogate":
        errors.append("manifest_condition_order_mismatch")
    if protocol.get("decoding") != {
        "do_sample": False,
        "max_new_tokens": 512,
        "num_beams": 1,
    }:
        errors.append("manifest_decoding_mismatch")

    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get("question_id"))].append(row)
    expected_ids = [
        str(question_id)
        for question_id in manifest.get("question_ids", [])
    ]
    if set(groups) != set(expected_ids):
        errors.append("raw_question_ids_do_not_match_manifest")
    if len(expected_ids) != len(set(expected_ids)):
        errors.append("manifest_question_ids_not_unique")

    condition_counts = Counter()
    episode_set = set()
    for question_id in expected_ids:
        group = groups.get(question_id, [])
        conditions = [row.get("condition") for row in group]
        condition_counts.update(conditions)
        if conditions != list(CONDITIONS):
            errors.append(
                f"{question_id}:conditions_or_order_mismatch"
            )
            continue
        if shared_projection(group[0]) != shared_projection(group[1]):
            errors.append(f"{question_id}:paired_input_mismatch")
        for row in group:
            condition = row["condition"]
            if row.get("forward_path") != FORWARD_PATHS[condition]:
                errors.append(f"{question_id}:{condition}:forward_path")
            if row.get("seed") != FROZEN_SEED:
                errors.append(f"{question_id}:{condition}:seed")
            if row.get("preparation_calls") != 1:
                errors.append(
                    f"{question_id}:{condition}:preparation_calls"
                )
            if row.get("prepared_prefill_cache_hits") != 2:
                errors.append(
                    f"{question_id}:{condition}:prepared_cache_hits"
                )
            expanded_hash = row.get("expanded_input_sha256")
            if (
                not isinstance(expanded_hash, str)
                or len(expanded_hash) != 64
            ):
                errors.append(
                    f"{question_id}:{condition}:expanded_input_hash"
                )
            full_tokens = row.get("full_tokens")
            spatial_start = row.get("spatial_start")
            visual_tokens = row.get("input_visual_tokens")
            if (
                not isinstance(full_tokens, int)
                or not isinstance(spatial_start, int)
                or not isinstance(visual_tokens, int)
                or visual_tokens < 1
                or not 0 <= spatial_start < full_tokens
            ):
                errors.append(
                    f"{question_id}:{condition}:spatial_span"
                )
            answer = row.get("answer")
            if not isinstance(answer, str):
                errors.append(f"{question_id}:{condition}:answer_type")
            output_hash = row.get("output_ids_sha256")
            if (
                not isinstance(output_hash, str)
                or len(output_hash) != 64
            ):
                errors.append(f"{question_id}:{condition}:output_hash")
            latency = row.get("latency_s")
            if (
                not isinstance(latency, (int, float))
                or not math.isfinite(latency)
                or latency < 0
            ):
                errors.append(f"{question_id}:{condition}:latency")
        episode_set.add(str(group[0].get("episode")))

    is_full = bool(manifest.get("is_full"))
    if require_full and not is_full:
        errors.append("manifest_not_full")
    if is_full or require_full:
        if len(expected_ids) != FROZEN_QUESTIONS:
            errors.append("full_question_count_mismatch")
        if len(episode_set) != FROZEN_EPISODES:
            errors.append("full_episode_count_mismatch")
        if len(rows) != FROZEN_QUESTIONS * len(CONDITIONS):
            errors.append("full_row_count_mismatch")

    provenance = manifest.get("provenance", {})
    required_sources = (
        "runner_sha256",
        "hm3d_adapter_sha256",
        "preflight_sha256",
        "audit_sha256",
        "scorer_sha256",
        "geogate_sha256",
        "scene_cap_sha256",
    )
    for field in required_sources:
        value = provenance.get(field)
        if not isinstance(value, str) or len(value) != 64:
            errors.append(f"missing_or_invalid_provenance:{field}")

    return {
        "state": "passed" if not errors else "failed",
        "errors": errors,
        "raw": {
            "path": str(raw_path),
            "sha256": digest_file(raw_path),
            "rows": len(rows),
            "questions": len(groups),
            "condition_counts": dict(sorted(condition_counts.items())),
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": digest_file(manifest_path),
            "is_full": is_full,
            "questions": len(expected_ids),
            "episodes": len(episode_set),
        },
        "scoring_authorized": not errors and is_full,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-full", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    result = audit(
        args.raw,
        args.manifest,
        require_full=args.require_full,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["state"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
