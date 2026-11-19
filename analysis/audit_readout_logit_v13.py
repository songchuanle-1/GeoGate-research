"""Independent structural audit for serialized v13 readout-logit rows."""

import argparse
import json
import math
from pathlib import Path

from analysis.readout_logit_v13_protocol import (
    BASELINE,
    arm_by_name,
    condition_names,
    delete_count,
)


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
    "baseline_token_id",
    "baseline_distribution_sha256",
)

METRIC_FIELDS = (
    "baseline_token_logprob",
    "baseline_token_margin",
    "js_divergence",
    "final_prompt_hidden_cosine",
    "latency_s",
)


def finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def audit_rows(rows, question_ids):
    expected_conditions = set(condition_names())
    expected_keys = {
        (str(question_id), condition)
        for question_id in question_ids
        for condition in expected_conditions
    }
    keys = [(str(row["question_id"]), row["condition"]) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected_keys:
        raise ValueError("Incomplete, duplicated, or unexpected v13 rows")

    arms = arm_by_name()
    references = {}
    for row in rows:
        question_id = str(row["question_id"])
        condition = row["condition"]
        if condition not in expected_conditions:
            raise ValueError("Unknown v13 condition")
        identity = {field: row[field] for field in IDENTITY_FIELDS}
        if question_id in references and identity != references[question_id]:
            raise ValueError("Input or baseline target changed within question group")
        references[question_id] = identity
        if any(not finite_number(row[field]) for field in METRIC_FIELDS):
            raise ValueError("Non-finite v13 metric")
        if not isinstance(row["argmax_token_id"], int):
            raise ValueError("Invalid argmax token")
        if not isinstance(row["argmax_flipped"], bool):
            raise ValueError("Invalid argmax-flip flag")
        if not isinstance(row["repeat_verified"], bool):
            raise ValueError("Invalid repeat-verification flag")
        if not 0.0 <= float(row["js_divergence"]) <= math.log(2.0) + 1e-6:
            raise ValueError("Jensen-Shannon divergence outside valid range")
        if not -1.000001 <= float(row["final_prompt_hidden_cosine"]) <= 1.000001:
            raise ValueError("Hidden-state cosine outside valid range")

        spatial = int(row["input_visual_tokens"])
        full = int(row["full_tokens"])
        start = int(row["spatial_start"])
        if not (spatial > 0 and 0 <= start and start + spatial <= full):
            raise ValueError("Invalid expanded spatial span")

        selected = row["selected_original_indices"]
        if selected != sorted(set(selected)):
            raise ValueError("Selected original indices are not sorted and unique")
        if any(index < 0 or index >= full for index in selected):
            raise ValueError("Selected original index outside input")
        spatial_end = start + spatial
        text = set(range(start)) | set(range(spatial_end, full))
        if not text.issubset(selected):
            raise ValueError("Text token removed")
        if (
            row["last_prompt_original_index"] != full - 1
            or row["final_output_original_index"] != full - 1
            or not row["prepared_embedding_verified_after_forward"]
            or row["use_cache"]
            or row["decode_forward_calls"] != 0
        ):
            raise ValueError("Invalid final-prompt or prefill-only invariant")

        if condition == BASELINE:
            if (
                row["layer"] is not None
                or row["ranking"] is not None
                or row["seed"] != 0
                or row["actual_deleted_spatial_tokens"] != 0
                or selected != list(range(full))
                or row["prefill_layer_input_lengths"] != [full] * 32
                or abs(float(row["baseline_token_logprob"])
                       - float(row["intervention_token_logprob"])) > 1e-10
                or row["argmax_flipped"]
                or row["forward_path"]
                != "native_unpatched_llama_forward"
                or row["baseline_forward_path"]
                != "native_unpatched_llama_forward"
            ):
                raise ValueError("Invalid baseline row")
            continue

        arm = arms[condition]
        if (
            row["layer"] != arm["layer"]
            or row["ranking"] != arm["ranking"]
            or row["seed"] != arm["seed"]
            or row["actual_deleted_spatial_tokens"] != delete_count(spatial)
            or row["forward_path"] != "geogate_intervention_forward"
            or row["baseline_forward_path"]
            != "native_unpatched_llama_forward"
        ):
            raise ValueError("Condition metadata or deletion budget mismatch")
        kept_spatial = sum(start <= index < spatial_end for index in selected)
        if kept_spatial != spatial - delete_count(spatial):
            raise ValueError("Observed retained spatial count is incorrect")
        if row["argmax_flipped"] != (
            row["argmax_token_id"] != row["baseline_token_id"]
        ):
            raise ValueError("Incorrect argmax-flip flag")
        expected_length = full - delete_count(spatial)
        expected_trace = (
            [full] * (arm["layer"] + 1)
            + [expected_length] * (31 - arm["layer"])
        )
        if row["prefill_layer_input_lengths"] != expected_trace:
            raise ValueError("Layer input lengths do not match intervention")

    return {
        "state": "passed",
        "questions": len(references),
        "conditions": len(expected_conditions),
        "rows": len(rows),
        "complete_groups": True,
        "shared_inputs": True,
        "prefill_only": True,
        "last_prompt_preserved": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = audit_rows(rows, manifest["question_ids"])
    repeat_required = bool(
        manifest.get("protocol", {}).get("repeat_verification", False)
    )
    if repeat_required and not all(
        row["repeat_verified"] for row in rows
    ):
        raise ValueError("Manifest requires repeated-forward verification")
    result["repeat_verification_required"] = repeat_required
    result["repeat_verification_passed"] = (
        not repeat_required
        or all(row["repeat_verified"] for row in rows)
    )
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
