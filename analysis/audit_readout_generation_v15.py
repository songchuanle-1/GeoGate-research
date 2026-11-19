#!/usr/bin/env python3
"""Independent structural audit for frozen v15 generation rows."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from analysis.readout_generation_v15_protocol import (
    BASELINE,
    MAX_NEW_TOKENS,
    arm_by_name,
    condition_names,
    delete_count,
)
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


def output_ids_sha256(output_ids):
    values = np.asarray(output_ids, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError("Output token IDs must be one-dimensional")
    return hashlib.sha256(values.tobytes()).hexdigest()


def finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def audit_rows(rows, question_ids, repeat_required=False):
    expected_conditions = condition_names()
    expected_keys = {
        (str(question_id), condition)
        for question_id in question_ids
        for condition in expected_conditions
    }
    keys = [
        (str(row["question_id"]), row["condition"])
        for row in rows
    ]
    if len(keys) != len(set(keys)) or set(keys) != expected_keys:
        raise ValueError("Incomplete, duplicated, or unexpected v15 rows")

    arms = arm_by_name()
    references = {}
    for row in rows:
        question_id = str(row["question_id"])
        condition = row["condition"]
        if condition not in expected_conditions:
            raise ValueError("Unknown v15 condition")
        identity = {field: row[field] for field in IDENTITY_FIELDS}
        if question_id in references and references[question_id] != identity:
            raise ValueError("Shared v15 input changed within a question")
        references[question_id] = identity

        if (
            row["condition_order"] != expected_conditions.index(condition)
            or row["prepared_prefill_cache_hit"] != row["condition_order"] + 1
            or row["preparation_calls"] != 1
            or row["condition_count"] != len(expected_conditions)
            or row["max_new_tokens"] != MAX_NEW_TOKENS
            or not row["prepared_embedding_verified_after_forward"]
        ):
            raise ValueError("Invalid v15 condition order or preparation reuse")
        expected_hits = len(expected_conditions) * (2 if repeat_required else 1)
        if row["final_prepared_prefill_cache_hits"] != expected_hits:
            raise ValueError("Incorrect final prepared-cache hit count")
        if row["repeat_verified"] != bool(repeat_required):
            raise ValueError("Repeat-verification metadata mismatch")
        if not finite_number(row["latency_s"]) or float(row["latency_s"]) < 0:
            raise ValueError("Invalid v15 latency")
        if (
            not isinstance(row["answer"], str)
            or not isinstance(row["output_ids"], list)
            or not row["output_ids"]
            or len(row["output_ids"]) > MAX_NEW_TOKENS
            or any(not isinstance(value, int) for value in row["output_ids"])
            or output_ids_sha256(row["output_ids"])
            != row["output_ids_sha256"]
        ):
            raise ValueError("Invalid or incorrectly hashed v15 output")
        if (
            not isinstance(row["decode_forward_calls"], int)
            or row["decode_forward_calls"] < 0
            or not row["decode_single_token_verified"]
            or not row["kv_all_32_layers_equal"]
            or len(row["complete_trace_sha256"]) != 64
        ):
            raise ValueError("Invalid v15 decode trace summary")
        if row["decode_forward_calls"] != len(
            row["kv_cache_lengths_by_forward"]
        ) - 1:
            raise ValueError("Decode count does not match KV trace")
        layer_matrix = row["layer_input_lengths_by_forward"]
        cache_matrix = row["kv_cache_lengths_by_forward"]
        if (
            len(layer_matrix) != len(cache_matrix)
            or any(len(lengths) != 32 for lengths in layer_matrix)
            or any(len(lengths) != 32 for lengths in cache_matrix)
            or trace_sha256(layer_matrix, cache_matrix)
            != row["complete_trace_sha256"]
        ):
            raise ValueError("Invalid or incorrectly hashed full v15 trace")

        spatial = int(row["input_visual_tokens"])
        full = int(row["full_tokens"])
        start = int(row["spatial_start"])
        if not (spatial > 0 and 0 <= start and start + spatial <= full):
            raise ValueError("Invalid expanded v15 spatial span")
        selected = row["selected_original_indices"]
        if selected != sorted(set(selected)):
            raise ValueError("Selected v15 indices are not sorted and unique")
        if any(index < 0 or index >= full for index in selected):
            raise ValueError("Selected v15 index outside input")
        spatial_end = start + spatial
        text_indices = set(range(start)) | set(range(spatial_end, full))
        if not text_indices.issubset(selected):
            raise ValueError("v15 removed a text token")
        if full - 1 not in selected:
            raise ValueError("v15 removed the final prompt token")

        if condition == BASELINE:
            if (
                row["layer"] is not None
                or row["ranking"] is not None
                or row["seed"] != 0
                or row["actual_deleted_spatial_tokens"] != 0
                or selected != list(range(full))
                or row["prefill_layer_input_lengths"] != [full] * 32
                or row["layer_input_lengths_by_forward"][1:]
                != [[1] * 32] * row["decode_forward_calls"]
                or row["kv_cache_length_summary"] != list(
                    range(full, full + row["decode_forward_calls"] + 1)
                )
                or row["kv_cache_lengths_by_forward"] != [
                    [length] * 32
                    for length in row["kv_cache_length_summary"]
                ]
                or row["forward_path"]
                != "native_unpatched_llama_forward"
            ):
                raise ValueError("Invalid v15 baseline row")
            continue

        arm = arms[condition]
        deleted = delete_count(spatial)
        reduced = full - deleted
        if (
            row["layer"] != arm["layer"]
            or row["ranking"] != arm["ranking"]
            or row["seed"] != arm["seed"]
            or row["actual_deleted_spatial_tokens"] != deleted
            or len(selected) != reduced
            or row["forward_path"] != "geogate_intervention_forward"
        ):
            raise ValueError("v15 intervention metadata or budget mismatch")
        kept_spatial = sum(start <= index < spatial_end for index in selected)
        if kept_spatial != spatial - deleted:
            raise ValueError("Incorrect v15 retained spatial count")
        expected_prefill = (
            [full] * (arm["layer"] + 1)
            + [reduced] * (31 - arm["layer"])
        )
        if row["prefill_layer_input_lengths"] != expected_prefill:
            raise ValueError("Incorrect v15 prefill layer lengths")
        if (
            row["layer_input_lengths_by_forward"][0] != expected_prefill
            or row["layer_input_lengths_by_forward"][1:]
            != [[1] * 32] * row["decode_forward_calls"]
        ):
            raise ValueError("Incorrect complete v15 layer trace")
        if row["kv_cache_length_summary"] != list(
            range(reduced, reduced + row["decode_forward_calls"] + 1)
        ):
            raise ValueError("Incorrect v15 KV cache growth")
        if row["kv_cache_lengths_by_forward"] != [
            [length] * 32
            for length in row["kv_cache_length_summary"]
        ]:
            raise ValueError("Incorrect complete v15 KV cache matrix")

    return {
        "state": "passed",
        "questions": len(references),
        "conditions": len(expected_conditions),
        "rows": len(rows),
        "complete_groups": True,
        "shared_prepared_inputs": True,
        "text_tokens_preserved": True,
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
    repeat_required = bool(
        manifest["protocol"].get("repeat_verification", False)
    )
    result = audit_rows(
        rows,
        manifest["question_ids"],
        repeat_required=repeat_required,
    )
    result["raw_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    result["manifest_sha256"] = hashlib.sha256(
        args.manifest.read_bytes()
    ).hexdigest()
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
