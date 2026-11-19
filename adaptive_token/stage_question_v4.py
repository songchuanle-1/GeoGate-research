"""One fully paired stage question; caller owns model/data provenance and seeds."""

from analysis.stage_protocol_v4 import arms, token_counts
from analysis.stage_trace_v4 import validate_stage_trace


def execute_question(question_id, scene_id, mode, question_index, generate_arm,
                     journal):
    """generate_arm must return predictions, actual hashes and raw decoder traces.

    No partial group is committed. Do not reconstruct a failed arm from scores.
    """
    conditions = arms(mode)
    offset = question_index % len(conditions)
    conditions = conditions[offset:] + conditions[:offset]
    rows = []
    reference = None
    for order, arm in enumerate(conditions):
        row = dict(generate_arm(arm))
        for key, value in (("question_id", str(question_id)), ("scene_id", scene_id),
                           ("condition", arm["name"]), ("seed", arm["seed"])):
            if row.get(key) != value:
                raise ValueError(f"Arm returned mismatched {key}")
        identity = {key: row[key] for key in (
            "input_tensor_hashes", "expanded_input_sha256", "prompt",
            "full_tokens", "spatial_start", "input_visual_tokens")}
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError("Input changed between stage arms")
        keep = token_counts(row["input_visual_tokens"], mode)["keep"]
        if row["actual_kept_spatial_tokens"] != keep:
            raise ValueError("Incorrect stage budget")
        validate_stage_trace(
            row["layer_trace"], row["kv_trace"], row["full_tokens"],
            row["spatial_start"], row["input_visual_tokens"], keep,
            arm["layer"], row["selected_original_indices"])
        row["order"] = order
        rows.append(row)
    journal.commit(str(question_id), rows)
    return len(rows)
