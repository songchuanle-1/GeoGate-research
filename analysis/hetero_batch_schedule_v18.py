"""Deterministic mixed-scene schedules for heterogeneous GeoGate batching."""

from collections import OrderedDict, deque
import random

from analysis.batch_throughput_v7_common import validate_inventory


WORKLOADS = ("mixed", "length", "worst")


def _question_rows(questions, inventory):
    grouped, recorded = validate_inventory(questions, inventory)
    rows = []
    for scene_id, scene_questions in grouped.items():
        info = recorded[scene_id]
        for scene_row_index, row in enumerate(scene_questions):
            rows.append({
                "question_id": str(row["question_id"]),
                "question_index": int(row["_question_index"]),
                "scene_id": str(scene_id),
                "scene_seed": int(info["scene_seed"]),
                "visual_tokens": int(info["visual_tokens"]),
                "expanded_valid_tokens": int(info["expanded_valid_tokens"]),
                "input_tokens": int(info["input_tokens"]),
                "scene_row_index": scene_row_index,
                "representative_question_id": str(
                    info["representative_question_id"]
                ),
            })
    return rows


def _scenes_in_order(rows):
    return list(OrderedDict((row["scene_id"], None) for row in rows))


def _batch_from_rows(name, batch_size, rows):
    scene_order = _scenes_in_order(rows)
    real_count = len(rows)
    slots = []
    for row in rows:
        slot = dict(row)
        slot["is_padding"] = False
        slot["source_question_id"] = str(row["question_id"])
        slots.append(slot)
    while len(slots) < batch_size:
        source = rows[-1]
        slot = dict(source)
        slot["is_padding"] = True
        slot["source_question_id"] = str(source["question_id"])
        slots.append(slot)
    visual = [row["visual_tokens"] for row in rows]
    return {
        "workload": name,
        "batch_size": int(batch_size),
        "real_count": real_count,
        "padding_count": int(batch_size) - real_count,
        "scene_count": len(scene_order),
        "cross_scene": len(scene_order) > 1,
        "visual_min": min(visual),
        "visual_max": max(visual),
        "visual_gap": max(visual) - min(visual),
        "slots": slots,
    }


def _mixed_rows(rows, seed):
    grouped = OrderedDict()
    for row in rows:
        grouped.setdefault(row["scene_id"], []).append(row)
    generator = random.Random(int(seed))
    scene_order = list(grouped)
    generator.shuffle(scene_order)
    for scene_id in scene_order:
        generator.shuffle(grouped[scene_id])

    ordered = []
    round_index = 0
    while any(grouped.values()):
        start = round_index % len(scene_order)
        rotated = scene_order[start:] + scene_order[:start]
        for scene_id in rotated:
            if grouped[scene_id]:
                ordered.append(grouped[scene_id].pop())
        round_index += 1
    return ordered


def _length_rows(rows):
    return sorted(
        rows,
        key=lambda row: (
            row["expanded_valid_tokens"],
            row["question_index"],
        ),
    )


def _worst_rows(rows):
    ordered = sorted(
        rows,
        key=lambda row: (
            row["expanded_valid_tokens"],
            row["question_index"],
        ),
    )
    pool = deque(ordered)
    result = []
    while pool:
        result.append(pool.popleft())
        if pool:
            result.append(pool.pop())
    return result


def build_workload_schedule(
    questions,
    inventory,
    workload,
    batch_size,
    shuffle_seed,
):
    """Build one deterministic schedule without dropping real questions."""
    if workload not in WORKLOADS:
        raise ValueError(f"Unsupported workload: {workload}")
    if int(batch_size) < 2:
        raise ValueError("Batch size must be at least two")
    rows = _question_rows(questions, inventory)
    if workload == "mixed":
        ordered = _mixed_rows(rows, shuffle_seed)
    elif workload == "length":
        ordered = _length_rows(rows)
    else:
        ordered = _worst_rows(rows)
    batches = []
    for batch_index, start in enumerate(
        range(0, len(ordered), int(batch_size))
    ):
        batch = _batch_from_rows(
            workload,
            batch_size,
            ordered[start:start + int(batch_size)],
        )
        batch["batch_index"] = batch_index
        batches.append(batch)
    return batches


def schedule_summary(batches):
    return {
        "workload": batches[0]["workload"] if batches else None,
        "batch_size": batches[0]["batch_size"] if batches else None,
        "batches": len(batches),
        "real_questions": sum(row["real_count"] for row in batches),
        "padding_slots": sum(row["padding_count"] for row in batches),
        "executed_slots": sum(
            row["batch_size"] for row in batches
        ),
        "cross_scene_batch_fraction": (
            sum(bool(row["cross_scene"]) for row in batches) / len(batches)
            if batches else 0.0
        ),
        "mean_visual_gap": (
            sum(row["visual_gap"] for row in batches) / len(batches)
            if batches else 0.0
        ),
        "max_visual_gap": (
            max(row["visual_gap"] for row in batches)
            if batches else 0
        ),
    }


def condition_order(batch_index, session_index, workload_index):
    phase = (
        int(batch_index) + int(session_index) + int(workload_index)
    ) % 2
    return (
        ("baseline", "geogate")
        if phase == 0 else
        ("geogate", "baseline")
    )
