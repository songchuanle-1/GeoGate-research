"""Frozen conditions for same-token-set cross-layer deletion."""

import hashlib
import struct


BASELINE = "baseline"
LAYERS = (8, 14, 20)
# Match the deterministic top-set arms in v15 (`seed=0`).  The fixed
# deletion set itself is not random; this only keeps the RNG state identical
# to the frozen source conditions.
FROZEN_SEED = 0


def digest_indices(indices) -> str:
    payload = b"".join(
        struct.pack("<I", int(index))
        for index in indices
    )
    return hashlib.sha256(payload).hexdigest()


def expected_selected(full_tokens, deleted_indices):
    deleted = {int(index) for index in deleted_indices}
    if len(deleted) != len(deleted_indices):
        raise ValueError("Deleted indices must be unique")
    selected = [
        index for index in range(int(full_tokens))
        if index not in deleted
    ]
    if not selected or selected[-1] != int(full_tokens) - 1:
        raise ValueError("Deletion must preserve the final prompt token")
    return selected


def condition_name(source_layer, prune_layer):
    if source_layer not in LAYERS or prune_layer not in LAYERS:
        raise ValueError("Source and prune layers must be in {8,14,20}")
    return f"fixedset_L{source_layer}_at_L{prune_layer}"


def condition_names():
    return [BASELINE] + [
        condition_name(source_layer, prune_layer)
        for source_layer in LAYERS
        for prune_layer in LAYERS
    ]


def arm_by_name():
    return {
        condition_name(source_layer, prune_layer): {
            "name": condition_name(source_layer, prune_layer),
            "source_layer": source_layer,
            "prune_layer": prune_layer,
        }
        for source_layer in LAYERS
        for prune_layer in LAYERS
    }


def primary_location_contrasts():
    """Damage differences under a fixed source set, comparing prune layers."""
    result = {}
    for source_layer in LAYERS:
        result[f"L{source_layer}set_at_L14_minus_at_L8"] = (
            condition_name(source_layer, 14),
            condition_name(source_layer, 8),
        )
        result[f"L{source_layer}set_at_L14_minus_at_L20"] = (
            condition_name(source_layer, 14),
            condition_name(source_layer, 20),
        )
    return result


def secondary_set_contrasts():
    """Damage differences under a fixed prune layer, comparing source sets."""
    result = {}
    for prune_layer in LAYERS:
        result[f"at_L{prune_layer}_L14set_minus_L8set"] = (
            condition_name(14, prune_layer),
            condition_name(8, prune_layer),
        )
        result[f"at_L{prune_layer}_L14set_minus_L20set"] = (
            condition_name(14, prune_layer),
            condition_name(20, prune_layer),
        )
    return result
