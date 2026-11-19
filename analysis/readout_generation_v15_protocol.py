"""Frozen arms and answer-level contrasts for LLaVA-3D v15."""

LAYERS = (8, 14, 20)
RANDOM_SEEDS = (101, 202, 303)
RANKINGS = ("top", "random", "bottom")
BASELINE = "baseline"
DELETE_FRACTION = 0.10
MAX_NEW_TOKENS = 512
FROZEN_SEED = 20260913


def balanced_scene_shards(inventory, num_shards):
    if num_shards < 1:
        raise ValueError("Number of shards must be positive")
    loads = [0] * num_shards
    shards = [[] for _ in range(num_shards)]
    ordered = sorted(
        inventory["scene_rows"],
        key=lambda row: (
            -int(row["question_count"])
            * int(row["expanded_valid_tokens"]) ** 2,
            int(row["scene_index"]),
        ),
    )
    for row in ordered:
        index = min(range(num_shards), key=lambda value: (loads[value], value))
        weight = (
            int(row["question_count"])
            * int(row["expanded_valid_tokens"]) ** 2
        )
        loads[index] += weight
        shards[index].append(row["scene_id"])
    official_index = {
        row["scene_id"]: int(row["scene_index"])
        for row in inventory["scene_rows"]
    }
    for shard in shards:
        shard.sort(key=official_index.__getitem__)
    return shards, loads


def intervention_arms():
    result = []
    for layer in LAYERS:
        for ranking in RANKINGS:
            seeds = RANDOM_SEEDS if ranking == "random" else (0,)
            for seed in seeds:
                result.append({
                    "name": f"delete10_L{layer}_{ranking}_s{seed}",
                    "layer": layer,
                    "ranking": ranking,
                    "seed": seed,
                    "scorer": {
                        "top": "attn",
                        "random": "random",
                        "bottom": "inverse_attn",
                    }[ranking],
                    "prune_mode": "delete_topk",
                })
    return result


def condition_names():
    return [BASELINE] + [arm["name"] for arm in intervention_arms()]


def arm_by_name():
    return {arm["name"]: arm for arm in intervention_arms()}


def delete_count(spatial_tokens):
    if not isinstance(spatial_tokens, int) or spatial_tokens < 1:
        raise ValueError("Positive integer spatial token count required")
    return min(
        spatial_tokens - 1,
        max(1, round(DELETE_FRACTION * spatial_tokens)),
    )


def primary_contrasts(quality):
    """Return layer-specific extra damage from deleting top-ranked tokens."""
    expected = {
        (layer, ranking, seed)
        for layer in LAYERS
        for ranking in RANKINGS
        for seed in (RANDOM_SEEDS if ranking == "random" else (0,))
    }
    if set(quality) != expected:
        raise ValueError("Incomplete or extra v15 quality arrays")
    damage = {}
    for layer in LAYERS:
        random_quality = sum(
            quality[(layer, "random", seed)]
            for seed in RANDOM_SEEDS
        ) / len(RANDOM_SEEDS)
        damage[layer] = random_quality - quality[(layer, "top", 0)]
    return {
        "L14_minus_L8": damage[14] - damage[8],
        "L14_minus_L20": damage[14] - damage[20],
    }


def top_over_bottom_damage(quality, layer):
    return (
        quality[(layer, "bottom", 0)]
        - quality[(layer, "top", 0)]
    )
