#!/usr/bin/env python3
"""Pure helpers for the frozen full batched-throughput protocol."""

from collections import OrderedDict


def scene_rows_in_official_order(questions):
    grouped = OrderedDict()
    for question_index, source in enumerate(questions):
        row = dict(source)
        row["_question_index"] = question_index
        grouped.setdefault(str(row["video"]), []).append(row)
    return grouped


def inventory_by_scene(inventory):
    return {
        str(row["scene_id"]): dict(row)
        for row in inventory["scene_rows"]
    }


def validate_inventory(questions, inventory):
    grouped = scene_rows_in_official_order(questions)
    recorded = inventory_by_scene(inventory)
    errors = []
    if list(grouped) != [str(row["scene_id"]) for row in inventory["scene_rows"]]:
        errors.append("inventory scene order differs from question-file order")
    for scene_id, rows in grouped.items():
        if scene_id not in recorded:
            errors.append(f"scene missing from inventory: {scene_id}")
            continue
        if int(recorded[scene_id]["question_count"]) != len(rows):
            errors.append(f"question count differs for {scene_id}")
    extra = sorted(set(recorded) - set(grouped))
    if extra:
        errors.append(f"inventory contains extra scenes: {extra}")
    if errors:
        raise ValueError("; ".join(errors))
    return grouped, recorded


def build_scene_batches(
    questions,
    inventory,
    batch_size,
    only_scenes=None,
):
    """Build fixed scene-only batches and duplicate the final row as padding."""
    grouped, recorded = validate_inventory(questions, inventory)
    allowed = set(map(str, only_scenes)) if only_scenes else None
    batches = []
    global_batch_index = 0
    for scene_index, (scene_id, rows) in enumerate(grouped.items()):
        if allowed is not None and scene_id not in allowed:
            continue
        scene_seed = int(recorded[scene_id]["scene_seed"])
        visual_tokens = int(recorded[scene_id]["visual_tokens"])
        for scene_batch_index, start in enumerate(range(0, len(rows), batch_size)):
            real_rows = rows[start:start + batch_size]
            slots = []
            for sample_index, row in enumerate(real_rows):
                item = dict(row)
                item["_is_padding"] = False
                item["_sample_index"] = sample_index
                item["_source_question_id"] = str(row["question_id"])
                slots.append(item)
            while len(slots) < batch_size:
                item = dict(real_rows[-1])
                item["_is_padding"] = True
                item["_sample_index"] = len(slots)
                item["_source_question_id"] = str(real_rows[-1]["question_id"])
                slots.append(item)
            batches.append({
                "batch_size": int(batch_size),
                "batch_index": global_batch_index,
                "scene_batch_index": scene_batch_index,
                "scene_index": scene_index,
                "scene_id": scene_id,
                "scene_seed": scene_seed,
                "visual_tokens": visual_tokens,
                "real_count": len(real_rows),
                "padding_count": batch_size - len(real_rows),
                "slots": slots,
            })
            global_batch_index += 1
    if allowed is not None:
        missing = sorted(allowed - {batch["scene_id"] for batch in batches})
        if missing:
            raise ValueError(f"Requested scenes not found: {missing}")
    return batches


def schedule_summary(batches):
    scenes = []
    seen = set()
    real_questions = 0
    padding_slots = 0
    for batch in batches:
        if batch["scene_id"] not in seen:
            scenes.append(batch["scene_id"])
            seen.add(batch["scene_id"])
        real_questions += int(batch["real_count"])
        padding_slots += int(batch["padding_count"])
    return {
        "batch_size": int(batches[0]["batch_size"]) if batches else None,
        "scenes": len(scenes),
        "batches": len(batches),
        "real_questions": real_questions,
        "padding_slots": padding_slots,
        "executed_slots": real_questions + padding_slots,
        "padding_fraction_of_executed": (
            padding_slots / (real_questions + padding_slots)
            if batches
            else 0.0
        ),
    }


def condition_order(batch_index, batch_size, session_index):
    """Counterbalance conditions across batches, sizes, and sessions."""
    phase = (
        int(batch_index)
        + int(session_index)
        + (0 if int(batch_size) == 2 else 1)
    ) % 2
    return (
        ("baseline", "geogate")
        if phase == 0
        else ("geogate", "baseline")
    )


def batch_size_order(session_index):
    return (4, 2) if int(session_index) % 2 == 0 else (2, 4)


def trim_after_eos(token_ids, eos_id=2, pad_id=0):
    values = list(token_ids)
    if eos_id in values:
        return values[:values.index(eos_id) + 1]
    while values and values[-1] == pad_id:
        values.pop()
    return values
