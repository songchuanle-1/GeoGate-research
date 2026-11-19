#!/usr/bin/env python3
"""Audit v18 throughput, exempting only the scheduled terminal remainder."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis.hetero_batch_schedule_v18 import condition_order


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def bootstrap_throughput_ratio(
    baseline,
    geogate,
    scenes,
    seed,
    iterations=10000,
):
    baseline = np.asarray(baseline, dtype=np.float64)
    geogate = np.asarray(geogate, dtype=np.float64)
    scenes = np.asarray(scenes)
    unique = np.asarray(list(dict.fromkeys(scenes.tolist())))
    index = {
        scene: np.flatnonzero(scenes == scene)
        for scene in unique
    }
    rng = np.random.default_rng(seed)
    results = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([index[scene] for scene in chosen])
        results[iteration] = (
            baseline[indices].sum() / geogate[indices].sum()
        )
    return [
        float(np.quantile(results, 0.025)),
        float(np.quantile(results, 0.975)),
    ]


def audit(raw_path, manifest_path, status_path):
    rows = load_jsonl(raw_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    errors = []
    if status.get("state") != "finished":
        errors.append("status is not finished")
    if status.get("raw_sha256") != digest_file(raw_path):
        errors.append("raw SHA256 mismatch")
    if status.get("manifest_sha256") != digest_file(manifest_path):
        errors.append("manifest SHA256 mismatch")
    schedule = manifest["protocol"]["schedule_summary"]
    scheduled_batches = int(schedule["batches"])
    scheduled_batch_size = int(schedule["batch_size"])
    scheduled_real = int(schedule["real_questions"])
    scheduled_padding = int(schedule["padding_slots"])
    if (
        scheduled_real
        != (
            (scheduled_batches - 1) * scheduled_batch_size
            + scheduled_batch_size - 1
        )
        or scheduled_padding != 1
    ):
        errors.append("v18c requires exactly one terminal remainder slot")
    terminal_batch_index = scheduled_batches - 1
    expected_rows = 2 * int(schedule["executed_slots"])
    if len(rows) != expected_rows:
        errors.append(f"expected {expected_rows} rows, found {len(rows)}")

    batches = defaultdict(dict)
    question_rows = {}
    for index, row in enumerate(rows):
        key = (
            int(row["batch_size"]),
            int(row["batch_index"]),
            str(row["condition"]),
            int(row["sample_index"]),
        )
        if key in batches:
            errors.append(f"duplicate slot: {key}")
        batches[key] = row
        if not row.get("is_padding"):
            prediction_key = (
                str(row["condition"]),
                str(row["question_id"]),
            )
            if prediction_key in question_rows:
                errors.append(f"duplicate prediction: {prediction_key}")
            question_rows[prediction_key] = row
        if float(row["peak_allocated_mib"]) < float(
            row["allocated_before_mib"]
        ):
            errors.append(f"peak below resident allocation: {key}")

    session_index = int(manifest["session_index"])
    terminal_remainder_count = 0
    for (batch_size, batch_index), group in _batch_groups(batches):
        for condition in ("baseline", "geogate"):
            slots = group.get(condition)
            if slots is None or set(slots) != set(range(batch_size)):
                errors.append(
                    f"B{batch_size} batch {batch_index} {condition}: "
                    "slot indices are incomplete"
                )
                continue
            batch_rows = [slots[index] for index in range(batch_size)]
            order = condition_order(batch_index, session_index, 0)
            expected_order = order.index(condition)
            for field in (
                "elapsed_batch_s",
                "order",
                "allocated_before_mib",
                "peak_allocated_mib",
                "incremental_peak_mib",
                "real_questions_in_batch",
                "padding_slots_in_batch",
                "scene_count_in_batch",
                "visual_gap_in_batch",
                "useful_generated_tokens_in_batch",
                "input_tensor_sha256",
            ):
                values = {
                    json.dumps(row[field], sort_keys=True)
                    for row in batch_rows
                }
                if len(values) != 1:
                    errors.append(
                        f"B{batch_size} batch {batch_index} {condition}: "
                        f"inconsistent {field}"
                    )
            if batch_rows[0]["order"] != expected_order:
                errors.append(
                    f"B{batch_size} batch {batch_index} {condition}: "
                    "condition order mismatch"
                )
            if condition == "baseline":
                if any(row["geogate_trace"] is not None for row in batch_rows):
                    errors.append("baseline contains GeoGate trace")
            else:
                traces = {
                    json.dumps(row["geogate_trace"], sort_keys=True)
                    for row in batch_rows
                }
                if len(traces) != 1:
                    errors.append("GeoGate trace differs across batch slots")
                trace = batch_rows[0]["geogate_trace"]
                if not isinstance(trace, list) or len(trace) != 1:
                    errors.append("invalid GeoGate trace length")
                else:
                    event = trace[0]
                    expected_visual = [
                        int(row["visual_tokens"])
                        for row in batch_rows
                    ]
                    if event.get("layer") != 14:
                        errors.append("wrong pruning layer")
                    if list(event.get("visual_counts", [])) != expected_visual:
                        errors.append("visual-count trace mismatch")
                    scene_count = len({
                        str(row["scene_id"]) for row in batch_rows
                    })
                    terminal_remainder = (
                        batch_index == terminal_batch_index
                        and scene_count == 1
                        and len(set(expected_visual)) == 1
                        and int(batch_rows[0]["real_questions_in_batch"])
                        == scheduled_batch_size - 1
                        and int(batch_rows[0]["padding_slots_in_batch"]) == 1
                    )
                    if terminal_remainder and condition == "geogate":
                        terminal_remainder_count += 1
                    elif (
                        scene_count < 2
                        or len(set(expected_visual)) < 2
                    ):
                        errors.append(
                            "non-terminal batch is not cross-scene and "
                            "heterogeneous"
                        )
    if terminal_remainder_count != 1:
        errors.append(
            "expected one terminal remainder batch, found "
            f"{terminal_remainder_count}"
        )

    # Allocate each batch's elapsed time equally over its real slots.
    per_query = defaultdict(dict)
    for (batch_size, batch_index, condition, sample_index), row in batches.items():
        if row["is_padding"]:
            continue
        per_query[str(row["question_id"])][condition] = (
            float(row["elapsed_batch_s"])
            / int(row["real_questions_in_batch"])
        )
    if len(per_query) != int(schedule["real_questions"]):
        errors.append("real prediction count differs from full schedule")

    if errors:
        return {
            "state": "failed",
            "errors": errors,
            "rows": len(rows),
            "questions": len(per_query),
            "session_index": session_index,
            "workload": manifest["workload"],
            "batch_size": int(manifest["batch_size"]),
            "provenance": {
                "raw_sha256": digest_file(raw_path),
                "manifest_sha256": digest_file(manifest_path),
                "status_sha256": digest_file(status_path),
                "auditor_sha256": digest_file(__file__),
            },
        }

    question_ids = sorted(per_query)
    scenes = np.asarray([
        str(question_rows[("baseline", question_id)]["scene_id"])
        for question_id in question_ids
    ])
    baseline = np.asarray([
        per_query[q]["baseline"] for q in question_ids
    ])
    geogate = np.asarray([
        per_query[q]["geogate"] for q in question_ids
    ])
    throughput_ratio = float(baseline.sum() / geogate.sum())
    ci = bootstrap_throughput_ratio(
        baseline,
        geogate,
        scenes,
        int(manifest["protocol"]["seed"]) + 100 * session_index,
    )
    passed = not errors
    return {
        "state": "passed" if passed else "failed",
        "errors": errors,
        "rows": len(rows),
        "questions": len(per_query),
        "session_index": session_index,
        "workload": manifest["workload"],
        "batch_size": int(manifest["batch_size"]),
        "baseline_sec_per_question": float(np.median(baseline)),
        "geogate_sec_per_question": float(np.median(geogate)),
        "latency_ratio": float(1.0 / throughput_ratio),
        "throughput_ratio": throughput_ratio,
        "scene_bootstrap_95_ci_throughput": ci,
        "promotion_gate_passed": (
            throughput_ratio >= 1.03 and ci[0] > 1.0
        ),
        "allocation": (
            "each batch elapsed time was divided equally over its real slots"
        ),
        "provenance": {
            "raw_sha256": digest_file(raw_path),
            "manifest_sha256": digest_file(manifest_path),
            "status_sha256": digest_file(status_path),
            "auditor_sha256": digest_file(__file__),
        },
    }


def _batch_groups(batches):
    grouped = {}
    for (batch_size, batch_index, condition, _sample_index), row in batches.items():
        grouped.setdefault((batch_size, batch_index), {}).setdefault(
            condition,
            {},
        )[int(_sample_index)] = row
    return sorted(grouped.items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    payload = audit(args.raw, args.manifest, args.status)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["state"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
