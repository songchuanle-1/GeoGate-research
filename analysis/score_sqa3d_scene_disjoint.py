"""Score the frozen SQA3D scene-disjoint confirmation with official EM."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from analysis.score_mmscan_geogate import (
    answer_match,
    clean_answer,
    summarize_output_lengths,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_QUESTIONS = 1_749
EXPECTED_SCENES = 35
NONINFERIORITY_MARGIN = -1.0
ITERATIONS = 10_000
SEED = 20260920


def display_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_or_jsonl(path: Path) -> list[dict]:
    raw = Path(path).read_text(encoding="utf-8")
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError:
        rows = [
            json.loads(line)
            for line in raw.splitlines()
            if line.strip()
        ]
    if not isinstance(rows, list) or not all(
        isinstance(row, dict) for row in rows
    ):
        raise ValueError(f"Expected a JSON array or JSONL objects: {path}")
    return rows


def unique_rows(rows: list[dict], label: str) -> dict[str, dict]:
    keyed = {}
    for row in rows:
        question_id = str(row["question_id"])
        if question_id in keyed:
            raise ValueError(f"Duplicate question ID in {label}: {question_id}")
        keyed[question_id] = row
    return keyed


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = round(quantile * (len(ordered) - 1))
    return ordered[min(len(ordered) - 1, max(0, index))]


def scene_bootstrap(
    deltas: dict[str, float],
    scenes: dict[str, str],
    *,
    iterations: int,
    seed: int,
    expected_scenes: int = EXPECTED_SCENES,
) -> dict:
    if set(deltas) != set(scenes):
        raise ValueError("Deltas and scenes must cover the same question IDs")
    grouped: dict[str, list[str]] = defaultdict(list)
    for question_id, scene in scenes.items():
        grouped[scene].append(question_id)
    scene_ids = sorted(grouped)
    if len(scene_ids) != expected_scenes:
        raise ValueError(f"Expected {expected_scenes} scene clusters")
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        sampled = [scene_ids[rng.randrange(len(scene_ids))]
                   for _ in scene_ids]
        keys = [
            question_id
            for scene in sampled
            for question_id in grouped[scene]
        ]
        estimates.append(
            sum(deltas[question_id] for question_id in keys) / len(keys)
        )
    observed = sum(deltas.values()) / len(deltas)
    return {
        "iterations": iterations,
        "seed": seed,
        "mean": observed,
        "two_sided_95ci": [
            percentile(estimates, 0.025),
            percentile(estimates, 0.975),
        ],
        "one_sided_975_lower_bound": percentile(estimates, 0.025),
    }


def score(
    *,
    subset_path: Path,
    baseline_path: Path,
    candidate_path: Path,
    answer_path: Path,
    iterations: int = ITERATIONS,
    seed: int = SEED,
    expected_questions: int = EXPECTED_QUESTIONS,
    expected_scenes: int = EXPECTED_SCENES,
    schema: str = "geogate-sqa3d-scene-disjoint-score-v1",
    scope: str | None = None,
) -> dict:
    subset = load_json_or_jsonl(subset_path)
    subset_by_id = unique_rows(subset, "SQA3D subset")
    baseline = unique_rows(load_json_or_jsonl(baseline_path), "baseline")
    candidate = unique_rows(load_json_or_jsonl(candidate_path), "candidate")
    answers = unique_rows(load_json_or_jsonl(answer_path), "answers")
    expected_ids = set(subset_by_id)
    if expected_ids != set(baseline):
        raise ValueError("Baseline IDs do not match the frozen subset")
    if expected_ids != set(candidate):
        raise ValueError("Candidate IDs do not match the frozen subset")
    missing_answers = expected_ids - set(answers)
    if missing_answers:
        raise ValueError(
            "Frozen subset IDs are missing from the answer file: "
            f"{len(missing_answers)}"
        )
    if len(expected_ids) != expected_questions:
        raise ValueError(f"Expected {expected_questions} questions")
    scenes = {
        question_id: str(subset_by_id[question_id]["scene_id"])
        for question_id in expected_ids
    }
    if len(set(scenes.values())) != expected_scenes:
        raise ValueError(f"Expected {expected_scenes} scenes")

    em_deltas = {}
    type_deltas: dict[int, list[float]] = defaultdict(list)
    baseline_em = 0
    candidate_em = 0
    baseline_refined_em = 0
    candidate_refined_em = 0
    baseline_raw = []
    candidate_raw = []
    win = tie = loss = 0
    for question_id in sorted(expected_ids):
        if baseline[question_id].get("prompt") != candidate[question_id].get(
            "prompt"
        ):
            raise ValueError(f"Prompt mismatch for {question_id}")
        references = [
            clean_answer(value) for value in answers[question_id]["text"]
        ]
        baseline_match = answer_match(
            clean_answer(baseline[question_id]["text"]),
            references,
        )
        candidate_match = answer_match(
            clean_answer(candidate[question_id]["text"]),
            references,
        )
        delta = float(candidate_match[0] - baseline_match[0])
        baseline_em += baseline_match[0]
        candidate_em += candidate_match[0]
        baseline_refined_em += baseline_match[1]
        candidate_refined_em += candidate_match[1]
        em_deltas[question_id] = delta
        question_type = int(subset_by_id[question_id]["type"])
        type_deltas[question_type].append(delta)
        win += int(delta > 0)
        tie += int(delta == 0)
        loss += int(delta < 0)
        baseline_raw.append(str(baseline[question_id]["text"]))
        candidate_raw.append(str(candidate[question_id]["text"]))

    bootstrap = scene_bootstrap(
        em_deltas,
        scenes,
        iterations=iterations,
        seed=seed,
        expected_scenes=expected_scenes,
    )
    count = len(expected_ids)
    delta_em = (candidate_em - baseline_em) / count
    lower = bootstrap["one_sided_975_lower_bound"]
    lower_em_points = lower * 100.0
    passes = bool(
        lower_em_points > NONINFERIORITY_MARGIN
        and count == expected_questions
        and len(set(scenes.values())) == expected_scenes
    )
    type_metrics = {}
    for question_type in sorted(type_deltas):
        ids = [
            question_id
            for question_id, value in scenes.items()
            if int(subset_by_id[question_id]["type"]) == question_type
        ]
        type_metrics[str(question_type)] = {
            "questions": len(type_deltas[question_type]),
            "scenes": len({scenes[question_id] for question_id in ids}),
            "delta_em_points": (
                sum(type_deltas[question_type])
                / len(type_deltas[question_type])
                * 100.0
            ),
        }
    result = {
        "schema": schema,
        "state": "passed",
        "inputs": {
            "subset": {
                "path": display_path(subset_path),
                "sha256": digest_file(subset_path),
            },
            "baseline": {
                "path": display_path(baseline_path),
                "sha256": digest_file(baseline_path),
            },
            "candidate": {
                "path": display_path(candidate_path),
                "sha256": digest_file(candidate_path),
            },
            "answers": {
                "path": display_path(answer_path),
                "sha256": digest_file(answer_path),
            },
        },
        "counts": {
            "questions": count,
            "scenes": len(set(scenes.values())),
            "question_type_counts": dict(sorted(Counter(
                int(row["type"]) for row in subset
            ).items())),
        },
        "metric_protocol": {
            "source": "3D-LLaVA eval_sqa3d.py / LEO semantics",
            "clean_answer": "exact evaluator clean_answer",
            "em": "exact membership after clean_answer",
            "refined_em": "substring containment after clean_answer",
            "primary_metric": "EM",
        },
        "metrics_percent": {
            "baseline_em": baseline_em / count * 100.0,
            "candidate_em": candidate_em / count * 100.0,
            "delta_em_points": delta_em * 100.0,
            "baseline_refined_em": baseline_refined_em / count * 100.0,
            "candidate_refined_em": candidate_refined_em / count * 100.0,
            "delta_refined_em_points": (
                candidate_refined_em - baseline_refined_em
            ) / count * 100.0,
        },
        "paired_question_outcomes": {
            "win": win,
            "tie": tie,
            "loss": loss,
        },
        "scene_bootstrap": {
            **bootstrap,
            "mean": bootstrap["mean"] * 100.0,
            "two_sided_95ci": [
                value * 100.0 for value in bootstrap["two_sided_95ci"]
            ],
            "one_sided_975_lower_bound": lower_em_points,
        },
        "type_metrics": type_metrics,
        "output_lengths": {
            "baseline": summarize_output_lengths(baseline_raw),
            "candidate": summarize_output_lengths(candidate_raw),
        },
        "promotion_gate": {
            "primary_endpoint": "EM",
            "noninferiority_margin_em_points": NONINFERIORITY_MARGIN,
            "one_sided_975_lower_bound_em_points": lower_em_points,
            "passes": passes,
            "decision": (
                "supports_frozen_noninferiority"
                if passes
                else "does_not_pass_frozen_noninferiority"
            ),
            "paper_updated": False,
        },
    }
    if scope is not None:
        result["scope"] = scope
    return result


def render_report(
    result: dict,
    *,
    title: str = "SQA3D Scene-Disjoint Confirmation",
    boundary: str | None = None,
) -> str:
    metrics = result["metrics_percent"]
    interval = result["scene_bootstrap"]["two_sided_95ci"]
    gate = result["promotion_gate"]
    lines = [
        f"# {title}",
        "",
        "Date: 2026-09-19 (Asia/Shanghai)",
        "",
        "## Outcome",
        "",
        f"State: `{result['state']}`.",
        "",
        f"- Questions: `{result['counts']['questions']}`",
        f"- Scenes: `{result['counts']['scenes']}`",
        f"- Baseline EM: `{metrics['baseline_em']:.4f}`",
        f"- GeoGate EM: `{metrics['candidate_em']:.4f}`",
        f"- Paired EM change: `{metrics['delta_em_points']:+.4f}` points",
        f"- Scene-cluster 95% CI: "
        f"`[{interval[0]:+.4f}, {interval[1]:+.4f}]`",
        f"- Frozen one-sided 97.5% lower bound: "
        f"`{gate['one_sided_975_lower_bound_em_points']:+.4f}`",
        f"- Promotion decision: `{gate['decision']}`",
        "",
        "The promotion rule was frozen before inspecting these outputs: "
        "pass only if the paired scene-cluster bootstrap 97.5% lower bound "
        "is greater than `-1.0` EM points.",
        "",
        "## Boundary",
        "",
        (
            boundary
            if boundary is not None
            else (
                "The 35 selected scenes are disjoint from the SQA3D "
                "train/test files and from the 71 ScanQA validation scenes "
                "used during GeoGate development. This is a new "
                "benchmark-level confirmation, not a claim that the "
                "checkpoint had no broader training exposure. The margin is "
                "operational and is not a user-elicited task tolerance."
            )
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--expected-questions",
        type=int,
        default=EXPECTED_QUESTIONS,
    )
    parser.add_argument(
        "--expected-scenes",
        type=int,
        default=EXPECTED_SCENES,
    )
    parser.add_argument(
        "--schema",
        default="geogate-sqa3d-scene-disjoint-score-v1",
    )
    parser.add_argument("--scope", default=None)
    parser.add_argument(
        "--title",
        default="SQA3D Scene-Disjoint Confirmation",
    )
    parser.add_argument("--boundary", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.out, args.report):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    result = score(
        subset_path=args.subset.resolve(),
        baseline_path=args.baseline.resolve(),
        candidate_path=args.candidate.resolve(),
        answer_path=args.answers.resolve(),
        iterations=args.iterations,
        seed=args.seed,
        expected_questions=args.expected_questions,
        expected_scenes=args.expected_scenes,
        schema=args.schema,
        scope=args.scope,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(
        render_report(
            result,
            title=args.title,
            boundary=args.boundary,
        ),
        encoding="utf-8",
    )
    print(json.dumps(result["metrics_percent"], indent=2, sort_keys=True))
    print(
        json.dumps(
            result["promotion_gate"],
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
