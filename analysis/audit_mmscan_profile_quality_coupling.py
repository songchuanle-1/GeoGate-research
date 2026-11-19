#!/usr/bin/env python3
"""Post-hoc scene-level audit of MMScan profile-to-quality coupling."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

try:
    from analysis.score_mmscan_geogate import (
        answer_match,
        clean_answer,
        pair_rows,
    )
except ModuleNotFoundError:
    from score_mmscan_geogate import (
        answer_match,
        clean_answer,
        pair_rows,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = (
    ROOT
    / "results/raw/followup/mmscan_3rscan_confirmation_20260917"
    / "attention_profile_full234_spanfix_20260919.json"
)
DEFAULT_SUBSET = (
    ROOT
    / "results/raw/followup/mmscan_3rscan_confirmation_20260917/subset.json"
)
DEFAULT_RUN = (
    ROOT
    / "results/raw/followup/mmscan_3rscan_confirmation_20260917"
    / "run_20frames_v3_20260918"
)
DEFAULT_OUT = (
    ROOT
    / "results/raw/followup/"
    "mmscan_profile_quality_coupling_20260920.json"
)
DEFAULT_REPORT = (
    ROOT / "results/MMSCAN_PROFILE_QUALITY_COUPLING_20260920.md"
)

MIDDLE = range(2, 14)
LATE = range(14, 27)


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def rank_average(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while (
            end < len(order)
            and values[order[end]] == values[order[start]]
        ):
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Correlation inputs must be non-empty and paired")
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right)
    )
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_ss * right_ss)
    if denominator == 0:
        raise ValueError("Correlation is undefined for a constant input")
    return numerator / denominator


def spearman(left: list[float], right: list[float]) -> float:
    return pearson(rank_average(left), rank_average(right))


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = round(quantile * (len(ordered) - 1))
    return ordered[min(len(ordered) - 1, max(0, index))]


def bootstrap_interval(
    left: list[float],
    right: list[float],
    *,
    statistic,
    iterations: int,
    seed: int,
) -> list[float]:
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        indices = [rng.randrange(len(left)) for _ in range(len(left))]
        sampled_left = [left[index] for index in indices]
        sampled_right = [right[index] for index in indices]
        try:
            estimates.append(statistic(sampled_left, sampled_right))
        except ValueError:
            continue
    if not estimates:
        raise ValueError("Bootstrap produced no finite correlation estimates")
    if len(estimates) < iterations // 2:
        raise ValueError("Bootstrap produced too few finite estimates")
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def permutation_p_value(
    left: list[float],
    right: list[float],
    *,
    statistic,
    iterations: int,
    seed: int,
) -> float:
    observed = abs(statistic(left, right))
    shuffled = list(right)
    rng = random.Random(seed)
    exceedances = 0
    for _ in range(iterations):
        rng.shuffle(shuffled)
        if abs(statistic(left, shuffled)) >= observed:
            exceedances += 1
    return (exceedances + 1) / (iterations + 1)


def holm_adjust(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=lambda name: (values[name], name))
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for position, name in enumerate(ordered):
        candidate = min(1.0, (count - position) * values[name])
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def profile_endpoints(profile: dict) -> list[float]:
    fraction = profile["frac"]
    coverage = profile["top50"]
    return [
        mean([fraction[index] for index in LATE])
        - mean([fraction[index] for index in MIDDLE]),
        fraction[14] - fraction[13],
        mean([coverage[index] for index in MIDDLE])
        - mean([coverage[index] for index in LATE]),
        mean([profile["half_mass_tokens"][index] for index in LATE]),
        mean([profile["gini"][index] for index in LATE]),
    ]


def build_audit(
    *,
    profile_path: Path = DEFAULT_PROFILE,
    subset_path: Path = DEFAULT_SUBSET,
    run_dir: Path = DEFAULT_RUN,
    iterations: int = 10_000,
    bootstrap_seed: int = 20260920,
    permutation_seed: int = 20260921,
) -> dict:
    profile_document = json.loads(
        Path(profile_path).read_text(encoding="utf-8")
    )
    profiles = profile_document.get("per_sample")
    if profile_document.get("state") != "complete" or not isinstance(
        profiles, list
    ):
        raise ValueError("Frozen MMScan profile is not complete")

    profile_by_scene = {}
    for row in profiles:
        scene = str(row["video"])
        if scene in profile_by_scene:
            raise ValueError(f"Duplicate profile scene: {scene}")
        if any(
            len(row[key]) != 32
            for key in ("frac", "top50", "half_mass_tokens", "gini")
        ):
            raise ValueError(f"Profile scene lacks 32 layers: {scene}")
        profile_by_scene[scene] = row
    if len(profile_by_scene) != 234:
        raise ValueError("Profile does not cover 234 unique scenes")

    subset = json.loads(Path(subset_path).read_text(encoding="utf-8"))
    question_metadata = {
        str(row["prompt_id"]): row for row in subset
    }
    rows = pair_rows(
        Path(run_dir) / "baseline.jsonl",
        Path(run_dir) / "geogate.jsonl",
    )
    if set(question_metadata) != {row["question_id"] for row in rows}:
        raise ValueError("Paired predictions do not match the frozen subset")

    quality_deltas: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        references = [
            clean_answer(value) for value in row["references"]
        ]
        baseline = answer_match(
            clean_answer(row["baseline"]),
            references,
        )
        candidate = answer_match(
            clean_answer(row["candidate"]),
            references,
        )
        scene = str(question_metadata[row["question_id"]]["video"])
        quality_deltas[scene].append(candidate[0] - baseline[0])

    scenes = sorted(profile_by_scene)
    if set(scenes) != set(quality_deltas):
        missing_profile = sorted(set(quality_deltas) - set(scenes))
        missing_quality = sorted(set(scenes) - set(quality_deltas))
        raise ValueError(
            "Scene coverage mismatch: "
            f"missing_profile={missing_profile[:5]}, "
            f"missing_quality={missing_quality[:5]}"
        )
    quality = [mean(quality_deltas[scene]) for scene in scenes]
    raw_endpoints = {
        scene: profile_endpoints(profile_by_scene[scene])
        for scene in scenes
    }

    definitions = (
        (
            "late_minus_middle_attention_fraction",
            "Late-minus-middle spatial-attention fraction",
            0,
        ),
        (
            "l14_minus_l13_attention_fraction",
            "L14-minus-L13 attention-fraction jump",
            1,
        ),
        (
            "middle_minus_late_half_mass_coverage",
            "Middle-minus-late half-mass coverage",
            2,
        ),
        (
            "late_mean_half_mass_tokens",
            "Late-phase mean half-mass token count",
            3,
        ),
        (
            "late_mean_gini",
            "Late-phase mean received-attention Gini",
            4,
        ),
    )
    endpoint_results = {}
    raw_p_values = {}
    for name, label, index in definitions:
        values = [raw_endpoints[scene][index] for scene in scenes]
        pearson_value = pearson(values, quality)
        spearman_value = spearman(values, quality)
        pearson_p = permutation_p_value(
            values,
            quality,
            statistic=pearson,
            iterations=iterations,
            seed=permutation_seed,
        )
        spearman_p = permutation_p_value(
            values,
            quality,
            statistic=spearman,
            iterations=iterations,
            seed=permutation_seed + 1,
        )
        endpoint_results[name] = {
            "label": label,
            "pearson": pearson_value,
            "pearson_scene_bootstrap_95ci": bootstrap_interval(
                values,
                quality,
                statistic=pearson,
                iterations=iterations,
                seed=bootstrap_seed,
            ),
            "pearson_permutation_p": pearson_p,
            "spearman": spearman_value,
            "spearman_scene_bootstrap_95ci": bootstrap_interval(
                values,
                quality,
                statistic=spearman,
                iterations=iterations,
                seed=bootstrap_seed + 1,
            ),
            "spearman_permutation_p": spearman_p,
            "profile_min": min(values),
            "profile_max": max(values),
        }
        raw_p_values[f"{name}:pearson"] = pearson_p
        raw_p_values[f"{name}:spearman"] = spearman_p

    adjusted = holm_adjust(raw_p_values)
    for name, _, _ in definitions:
        endpoint_results[name]["pearson_holm_p"] = adjusted[
            f"{name}:pearson"
        ]
        endpoint_results[name]["spearman_holm_p"] = adjusted[
            f"{name}:spearman"
        ]

    primary = endpoint_results[
        "late_minus_middle_attention_fraction"
    ]
    primary_supported_positive = (
        primary["pearson"] > 0
        and primary["pearson_scene_bootstrap_95ci"][0] > 0
        and primary["spearman"] > 0
        and primary["spearman_scene_bootstrap_95ci"][0] > 0
    )
    primary_supported_negative = (
        primary["pearson"] < 0
        and primary["pearson_scene_bootstrap_95ci"][1] < 0
        and primary["spearman"] < 0
        and primary["spearman_scene_bootstrap_95ci"][1] < 0
    )
    if primary_supported_positive:
        decision = "supported_positive_monotonic_coupling"
    elif primary_supported_negative:
        decision = "supported_negative_monotonic_coupling"
    else:
        decision = "no_supported_monotonic_coupling"

    return {
        "schema": "geogate-mmscan-profile-quality-coupling-v1",
        "state": "complete",
        "scope": (
            "Post-hoc scene-level association between the frozen external "
            "attention profile and the frozen paired L14/R=0.10 EM change. "
            "The outputs were known before this audit was frozen. It is "
            "descriptive, not preregistered, causal, or confirmatory."
        ),
        "counts": {
            "questions": len(rows),
            "scenes": len(scenes),
            "scene_question_min": min(len(values) for values in quality_deltas.values()),
            "scene_question_max": max(len(values) for values in quality_deltas.values()),
            "scenes_with_nonzero_mean_delta": sum(
                value != 0 for value in quality
            ),
        },
        "method": {
            "quality": (
                "Scene-mean paired exact-match change, candidate minus "
                "baseline, using the frozen official clean-answer semantics."
            ),
            "profile_layers": {
                "middle": "L2--L13",
                "late": "L14--L26",
            },
            "bootstrap": (
                "10,000 scene resamples; percentile 95% interval."
            ),
            "permutation": (
                "10,000 profile-label permutations; two-sided absolute "
                "statistic with a plus-one correction."
            ),
            "multiplicity": "Holm adjustment across the five endpoints.",
            "decision_rule": (
                "The primary late-minus-middle endpoint supports monotonic "
                "coupling only when both Pearson and Spearman point "
                "estimates have the same sign and both scene-bootstrap "
                "95% intervals exclude zero in that direction."
            ),
        },
        "decision": decision,
        "primary_endpoint": "late_minus_middle_attention_fraction",
        "endpoints": endpoint_results,
        "quality_scene_mean": {
            "mean": mean(quality),
            "min": min(quality),
            "max": max(quality),
        },
        "hashes": {
            "profile": digest_file(profile_path),
            "subset": digest_file(subset_path),
            "baseline": digest_file(Path(run_dir) / "baseline.jsonl"),
            "geogate": digest_file(Path(run_dir) / "geogate.jsonl"),
        },
    }


def render_report(audit: dict) -> str:
    lines = [
        "# MMScan Profile-to-Quality Coupling Audit",
        "",
        "Date: 2026-09-20 (Asia/Shanghai)",
        "",
        "## Scope",
        "",
        audit["scope"],
        "",
        "The analysis uses all "
        f"`{audit['counts']['questions']}` questions and "
        f"`{audit['counts']['scenes']}` scenes from the frozen 3RScan run. "
        "The profile and prediction outputs were both already inspected "
        "before this analysis was specified; it is therefore post-hoc.",
        "",
        "## Outcome",
        "",
        f"Decision: `{audit['decision']}`.",
        "",
        "| Profile endpoint | Pearson (95% scene CI) | Spearman (95% scene CI) |",
        "|---|---:|---:|",
    ]
    for name in (
        "late_minus_middle_attention_fraction",
        "l14_minus_l13_attention_fraction",
        "middle_minus_late_half_mass_coverage",
        "late_mean_half_mass_tokens",
        "late_mean_gini",
    ):
        result = audit["endpoints"][name]
        pearson_ci = result["pearson_scene_bootstrap_95ci"]
        spearman_ci = result["spearman_scene_bootstrap_95ci"]
        lines.append(
            f"| {result['label']} | {result['pearson']:+.4f} "
            f"[{pearson_ci[0]:+.4f},{pearson_ci[1]:+.4f}] | "
            f"{result['spearman']:+.4f} "
            f"[{spearman_ci[0]:+.4f},{spearman_ci[1]:+.4f}] |"
        )
    lines.extend([
        "",
        "The primary late-minus-middle endpoint has a mean scene quality "
        f"change of {audit['quality_scene_mean']['mean']:+.4f} EM and "
        "neither Pearson nor Spearman coupling clears the frozen decision "
        "rule. The other four endpoints are secondary and Holm-adjusted.",
        "",
        "## Interpretation",
        "",
        "This result does not show that the external attention profile is "
        "unimportant. It shows that the frozen scene-level profile summaries "
        "do not provide a reproducible linear or rank explanation of the "
        "small paired quality changes in this run. The profile-persistence "
        "result and the inconclusive pruning-quality result therefore remain "
        "separate evidence: pattern persistence does not imply quality "
        "preservation, and the inconclusive quality interval is not rescued "
        "by profile-to-quality coupling.",
        "",
        "No paper headline is changed by this audit. It is a post-hoc "
        "negative diagnostic on the same already-inspected external run.",
        "",
        "## Hashes",
        "",
    ])
    for name in sorted(audit["hashes"]):
        digest = audit["hashes"][name]
        lines.append(f"- `{name}`: `{digest}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260920)
    parser.add_argument("--permutation-seed", type=int, default=20260921)
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite {args.report}")

    audit = build_audit(
        profile_path=args.profile,
        subset_path=args.subset,
        run_dir=args.run_dir,
        iterations=args.iterations,
        bootstrap_seed=args.bootstrap_seed,
        permutation_seed=args.permutation_seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(render_report(audit), encoding="utf-8")
    print(json.dumps({
        "decision": audit["decision"],
        "primary": audit["endpoints"][audit["primary_endpoint"]],
        "out": str(args.out),
        "report": str(args.report),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
