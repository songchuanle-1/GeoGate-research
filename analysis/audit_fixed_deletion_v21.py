#!/usr/bin/env python3
"""Independent structural audit for fixed-set cross-layer deletion rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from analysis.audit_readout_generation_v15 import output_ids_sha256
from analysis.readout_fixed_deletion_protocol import (
    BASELINE,
    FROZEN_SEED,
    arm_by_name,
    condition_names,
    digest_indices,
    expected_selected,
)
from analysis.readout_generation_v15_protocol import MAX_NEW_TOKENS
from analysis.readout_generation_v15_trace import trace_sha256


IDENTITY_FIELDS = (
    "scene_id",
    "prompt",
    "scene_seed",
    "input_tensor_hashes",
    "expanded_input_sha256",
    "full_tokens",
    "spatial_start",
    "input_visual_tokens",
    "deterministic_cap",
    "preparation_calls",
    "condition_count",
    "max_new_tokens",
)


def finite_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float("-inf") < float(value) < float("inf")
    )


def audit_rows(rows, question_ids, repeat_required=False):
    conditions = condition_names()
    expected_keys = {
        (str(question_id), condition)
        for question_id in question_ids
        for condition in conditions
    }
    keys = [
        (str(row["question_id"]), str(row["condition"]))
        for row in rows
    ]
    if len(keys) != len(set(keys)) or set(keys) != expected_keys:
        raise ValueError(
            "Incomplete, duplicated, or unexpected fixed-deletion rows"
        )

    references = {}
    groups = {}
    arms = arm_by_name()
    for row in rows:
        question_id = str(row["question_id"])
        condition = str(row["condition"])
        identity = {field: row[field] for field in IDENTITY_FIELDS}
        if question_id in references and references[question_id] != identity:
            raise ValueError(
                "Shared input changed within a fixed-deletion question"
            )
        references[question_id] = identity
        groups.setdefault(question_id, {})[condition] = row

        if (
            row["condition_order"] != conditions.index(condition)
            or row["prepared_prefill_cache_hit"]
            != row["condition_order"] + 1
            or row["preparation_calls"] != 1
            or row["condition_count"] != len(conditions)
            or row["max_new_tokens"] != MAX_NEW_TOKENS
            or not row["prepared_embedding_verified_after_forward"]
        ):
            raise ValueError(
                "Invalid condition order or prepared-input reuse"
            )
        expected_hits = len(conditions) * (2 if repeat_required else 1)
        if row["final_prepared_prefill_cache_hits"] != expected_hits:
            raise ValueError("Incorrect final prepared-cache hit count")
        if row["repeat_verified"] != bool(repeat_required):
            raise ValueError("Repeat-verification metadata mismatch")
        if not finite_number(row["latency_s"]) or float(row["latency_s"]) < 0:
            raise ValueError("Invalid generation latency")
        if (
            not isinstance(row["answer"], str)
            or not isinstance(row["output_ids"], list)
            or not row["output_ids"]
            or len(row["output_ids"]) > MAX_NEW_TOKENS
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in row["output_ids"]
            )
            or output_ids_sha256(row["output_ids"])
            != row["output_ids_sha256"]
        ):
            raise ValueError("Invalid or incorrectly hashed output")
        if (
            not isinstance(row["decode_forward_calls"], int)
            or row["decode_forward_calls"] < 0
            or not row["decode_single_token_verified"]
            or not row["kv_all_32_layers_equal"]
            or len(row["complete_trace_sha256"]) != 64
        ):
            raise ValueError("Invalid decode trace summary")
        if row["decode_forward_calls"] != len(
            row["kv_cache_lengths_by_forward"]
        ) - 1:
            raise ValueError("Decode count does not match KV trace")
        layer_matrix = row["layer_input_lengths_by_forward"]
        cache_matrix = row["kv_cache_lengths_by_forward"]
        if (
            len(layer_matrix) != len(cache_matrix)
            or any(len(values) != 32 for values in layer_matrix)
            or any(len(values) != 32 for values in cache_matrix)
            or trace_sha256(layer_matrix, cache_matrix)
            != row["complete_trace_sha256"]
        ):
            raise ValueError("Invalid or incorrectly hashed full trace")

        full = int(row["full_tokens"])
        start = int(row["spatial_start"])
        spatial = int(row["input_visual_tokens"])
        if not (full > 0 and spatial > 0 and 0 <= start):
            raise ValueError("Invalid spatial span")
        if start + spatial > full:
            raise ValueError("Spatial span exceeds full sequence")
        selected = row["selected_original_indices"]
        if (
            not isinstance(selected, list)
            or not selected
            or selected != sorted(set(selected))
            or any(index < 0 or index >= full for index in selected)
        ):
            raise ValueError("Selected indices are not sorted unique")
        text_indices = set(range(start)) | set(
            range(start + spatial, full)
        )
        if not text_indices.issubset(selected):
            raise ValueError("Fixed deletion removed a text token")
        if full - 1 not in selected:
            raise ValueError("Fixed deletion removed the final prompt token")

        if condition == BASELINE:
            if (
                row["layer"] is not None
                or row["source_layer"] is not None
                or row["ranking"] is not None
                or row["seed"] != 0
                or row["actual_deleted_spatial_tokens"] != 0
                or row["fixed_deleted_original_indices"] != []
                or row["fixed_deleted_sha256"] is not None
                or selected != list(range(full))
                or row["prefill_layer_input_lengths"] != [full] * 32
                or row["layer_input_lengths_by_forward"][1:]
                != [[1] * 32] * row["decode_forward_calls"]
                or row["kv_cache_length_summary"] != list(
                    range(full, full + row["decode_forward_calls"] + 1)
                )
                or row["kv_cache_lengths_by_forward"]
                != [
                    [length] * 32
                    for length in row["kv_cache_length_summary"]
                ]
                or row["forward_path"] != "native_unpatched_llama_forward"
            ):
                raise ValueError("Invalid fixed-deletion baseline row")
            continue

        arm = arms[condition]
        deleted = row["fixed_deleted_original_indices"]
        if (
            row["layer"] != arm["prune_layer"]
            or row["source_layer"] != arm["source_layer"]
            or row["ranking"]
            != f"fixed_top_from_L{arm['source_layer']}"
            or row["seed"] != FROZEN_SEED
            or not isinstance(deleted, list)
            or not deleted
            or deleted != sorted(set(deleted))
            or any(
                index < start or index >= start + spatial
                for index in deleted
            )
            or row["actual_deleted_spatial_tokens"] != len(deleted)
            or row["fixed_deleted_sha256"] != digest_indices(deleted)
            or selected != expected_selected(full, deleted)
            or row["forward_path"] != "geogate_fixed_deletion_forward"
        ):
            raise ValueError(
                "Fixed-deletion intervention metadata or set mismatch"
            )
        reduced = full - len(deleted)
        expected_prefill = (
            [full] * (arm["prune_layer"] + 1)
            + [reduced] * (31 - arm["prune_layer"])
        )
        if row["prefill_layer_input_lengths"] != expected_prefill:
            raise ValueError("Incorrect fixed-deletion prefill lengths")
        if (
            row["layer_input_lengths_by_forward"][0] != expected_prefill
            or row["layer_input_lengths_by_forward"][1:]
            != [[1] * 32] * row["decode_forward_calls"]
        ):
            raise ValueError("Incorrect complete fixed-deletion layer trace")
        if row["kv_cache_length_summary"] != list(
            range(reduced, reduced + row["decode_forward_calls"] + 1)
        ):
            raise ValueError("Incorrect fixed-deletion KV growth")
        if row["kv_cache_lengths_by_forward"] != [
            [length] * 32
            for length in row["kv_cache_length_summary"]
        ]:
            raise ValueError("Incorrect fixed-deletion KV matrix")

    for question_id, group in groups.items():
        for source_layer in (8, 14, 20):
            source_rows = [
                group[
                    f"fixedset_L{source_layer}_at_L{prune_layer}"
                ]
                for prune_layer in (8, 14, 20)
            ]
            sets = [
                row["fixed_deleted_original_indices"]
                for row in source_rows
            ]
            hashes = {
                row["fixed_deleted_sha256"] for row in source_rows
            }
            if any(value != sets[0] for value in sets) or len(hashes) != 1:
                raise ValueError(
                    f"{question_id}: source set changed across prune layers"
                )

    return {
        "state": "passed",
        "questions": len(references),
        "conditions": len(conditions),
        "rows": len(rows),
        "complete_groups": True,
        "shared_prepared_inputs": True,
        "text_tokens_preserved": True,
        "fixed_sets_reused_across_layers": True,
        "single_token_decode_verified": True,
        "kv_growth_verified": True,
        "repeat_verification_required": bool(repeat_required),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = audit_rows(
        rows,
        [str(value) for value in manifest["question_ids"]],
        repeat_required=bool(
            manifest["protocol"].get("repeat_verification", False)
        ),
    )
    result["raw_sha256"] = hashlib.sha256(
        args.input.read_bytes()
    ).hexdigest()
    result["manifest_sha256"] = hashlib.sha256(
        args.manifest.read_bytes()
    ).hexdigest()
    lock = manifest.get("experiment_lock")
    if (
        not isinstance(lock, dict)
        or lock.get("experiment") != "fixed_deletion_v21"
        or len(str(lock.get("sha256", ""))) != 64
    ):
        raise ValueError("Fixed-deletion manifest has no valid experiment lock")
    result["experiment_lock_sha256"] = str(lock["sha256"])
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
