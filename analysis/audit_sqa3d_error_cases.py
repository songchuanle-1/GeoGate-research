"""Classify paired SQA3D validation wins and losses on inspected folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from analysis.score_mmscan_geogate import answer_match, clean_answer
from analysis.score_sqa3d_scene_disjoint import (
    display_path,
    load_json_or_jsonl,
    unique_rows,
)


EXPECTED = {
    "validation_scene_disjoint": (1_749, 35),
    "validation_replication": (1_512, 30),
}
QUESTION_KINDS = (
    "count",
    "spatial",
    "color",
    "shape",
    "description",
    "other",
)


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def question_kind(text: str) -> str:
    normalized = " ".join(str(text).lower().split())
    if any(term in normalized for term in ("how many", "amount of", "number of")):
        return "count"
    if "shape" in normalized or any(
        term in normalized
        for term in ("round", "square", "rectangular", "triangle")
    ):
        return "shape"
    if "color" in normalized or "colour" in normalized:
        return "color"
    if any(
        term in normalized
        for term in (
            "where",
            "which side",
            "left",
            "right",
            "behind",
            "front",
            "above",
            "below",
            "under",
            "between",
            "closest",
            "direction",
            "turn",
        )
    ):
        return "spatial"
    if any(
        term in normalized
        for term in (
            "what is",
            "which object",
            "what object",
            "what color",
            "what shape",
            "what material",
        )
    ):
        return "description"
    return "other"


def response_length(text: str) -> int:
    return len(clean_answer(text).split())


def _load_fold(
    *,
    name: str,
    subset_path: Path,
    baseline_path: Path,
    candidate_path: Path,
    answers: dict[str, dict],
) -> dict:
    expected_questions, expected_scenes = EXPECTED[name]
    subset = unique_rows(load_json_or_jsonl(subset_path), f"{name} subset")
    baseline = unique_rows(load_json_or_jsonl(baseline_path), f"{name} baseline")
    candidate = unique_rows(
        load_json_or_jsonl(candidate_path),
        f"{name} candidate",
    )
    question_ids = set(subset)
    if len(question_ids) != expected_questions:
        raise ValueError(f"{name}: unexpected question count")
    if set(baseline) != question_ids or set(candidate) != question_ids:
        raise ValueError(f"{name}: prediction IDs do not match the subset")
    if len({str(row["scene_id"]) for row in subset.values()}) != expected_scenes:
        raise ValueError(f"{name}: unexpected scene count")
    missing_answers = question_ids - set(answers)
    if missing_answers:
        raise ValueError(f"{name}: missing {len(missing_answers)} answers")

    rows = []
    for question_id in sorted(question_ids):
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
        baseline_em, baseline_refined = baseline_match
        candidate_em, candidate_refined = candidate_match
        if baseline_em and candidate_em:
            outcome = "both_correct"
        elif not baseline_em and not candidate_em:
            outcome = "both_wrong"
        elif baseline_em and not candidate_em:
            outcome = "loss"
        else:
            outcome = "win"
        loss_kind = None
        if outcome == "loss":
            loss_kind = (
                "near_miss_refined"
                if candidate_refined
                else "hard_loss"
            )
        rows.append({
            "question_id": question_id,
            "scene_id": str(subset[question_id]["scene_id"]),
            "official_type": str(int(subset[question_id]["type"])),
            "question_kind": question_kind(subset[question_id]["text"]),
            "outcome": outcome,
            "loss_kind": loss_kind,
            "baseline_answer": clean_answer(baseline[question_id]["text"]),
            "candidate_answer": clean_answer(candidate[question_id]["text"]),
            "baseline_refined": bool(baseline_refined),
            "candidate_refined": bool(candidate_refined),
            "candidate_minus_baseline_words": (
                response_length(candidate[question_id]["text"])
                - response_length(baseline[question_id]["text"])
            ),
        })
    return {
        "name": name,
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
        },
        "rows": rows,
    }


def _count_by(rows: list[dict], key: str) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        grouped[str(row[key])][row["outcome"]] += 1
    return {
        group: {
            outcome: int(counts.get(outcome, 0))
            for outcome in (
                "win",
                "loss",
                "both_correct",
                "both_wrong",
            )
        }
        for group, counts in sorted(grouped.items())
    }


def _answer_pairs(rows: list[dict], outcome: str) -> list[dict]:
    counts = Counter(
        (row["baseline_answer"], row["candidate_answer"])
        for row in rows
        if row["outcome"] == outcome
    )
    return [
        {
            "baseline_answer": baseline,
            "candidate_answer": candidate,
            "questions": count,
        }
        for (baseline, candidate), count in counts.most_common(20)
    ]


def _fold_summary(fold: dict) -> dict:
    rows = fold["rows"]
    outcomes = Counter(row["outcome"] for row in rows)
    losses = [row for row in rows if row["outcome"] == "loss"]
    wins = [row for row in rows if row["outcome"] == "win"]
    return {
        "counts": {
            "questions": len(rows),
            "scenes": len({row["scene_id"] for row in rows}),
            "outcomes": dict(sorted(outcomes.items())),
            "loss_kind": dict(sorted(Counter(
                row["loss_kind"] for row in losses
            ).items())),
        },
        "by_official_type": _count_by(rows, "official_type"),
        "by_question_kind": _count_by(rows, "question_kind"),
        "top_loss_answer_pairs": _answer_pairs(rows, "loss"),
        "top_win_answer_pairs": _answer_pairs(rows, "win"),
        "loss_length_delta": {
            "candidate_shorter": sum(
                row["candidate_minus_baseline_words"] < 0 for row in losses
            ),
            "same_length": sum(
                row["candidate_minus_baseline_words"] == 0 for row in losses
            ),
            "candidate_longer": sum(
                row["candidate_minus_baseline_words"] > 0 for row in losses
            ),
        },
        "representative_losses": losses[:20],
        "representative_wins": wins[:20],
    }


def build_audit(
    *,
    fold_a_subset: Path,
    fold_a_baseline: Path,
    fold_a_candidate: Path,
    fold_b_subset: Path,
    fold_b_baseline: Path,
    fold_b_candidate: Path,
    answers_path: Path,
) -> dict:
    answers = unique_rows(load_json_or_jsonl(answers_path), "answers")
    folds = {
        "validation_scene_disjoint": _load_fold(
            name="validation_scene_disjoint",
            subset_path=fold_a_subset,
            baseline_path=fold_a_baseline,
            candidate_path=fold_a_candidate,
            answers=answers,
        ),
        "validation_replication": _load_fold(
            name="validation_replication",
            subset_path=fold_b_subset,
            baseline_path=fold_b_baseline,
            candidate_path=fold_b_candidate,
            answers=answers,
        ),
    }
    return {
        "schema": "geogate-sqa3d-error-case-audit-v1",
        "state": "passed",
        "scope": (
            "Post-hoc paired error-transition audit over already inspected "
            "SQA3D validation outputs. It does not rerun the model, identify "
            "token causality, or create an independent confirmation."
        ),
        "folds": {
            name: _fold_summary(fold)
            for name, fold in folds.items()
        },
        "inputs": {
            "answers": {
                "path": display_path(answers_path),
                "sha256": digest_file(answers_path),
            },
            **{
                f"{name}_{kind}": value
                for name, fold in folds.items()
                for kind, value in fold["inputs"].items()
            },
        },
        "boundary": {
            "paper_updated": False,
            "independent_confirmation": False,
            "post_hoc_case_selection": True,
        },
    }


def render_report(result: dict) -> str:
    lines = [
        "# SQA3D Error-Transition Audit",
        "",
        "Date: 2026-09-20 (Asia/Shanghai)",
        "",
        "Post-hoc paired analysis over the two inspected validation folds. "
        "It creates no new confirmation and does not authorize a paper edit.",
        "",
    ]
    for name, fold in result["folds"].items():
        lines.extend([
            f"## {name}",
            "",
            f"- Questions: `{fold['counts']['questions']}`",
            f"- Outcomes: `{fold['counts']['outcomes']}`",
            f"- Loss types: `{fold['counts']['loss_kind']}`",
            "",
            "| Official type | Wins | Losses | Both correct | Both wrong |",
            "|---:|---:|---:|---:|---:|",
        ])
        for key, counts in fold["by_official_type"].items():
            lines.append(
                f"| {key} | {counts['win']} | {counts['loss']} | "
                f"{counts['both_correct']} | {counts['both_wrong']} |"
            )
        lines.extend([
            "",
            "| Question kind | Wins | Losses | Both correct | Both wrong |",
            "|---|---:|---:|---:|---:|",
        ])
        for key, counts in fold["by_question_kind"].items():
            lines.append(
                f"| {key} | {counts['win']} | {counts['loss']} | "
                f"{counts['both_correct']} | {counts['both_wrong']} |"
            )
        lines.extend([
            "",
            f"- Loss answer lengths: `{fold['loss_length_delta']}`",
            "",
        ])
    lines.extend([
        "## Boundary",
        "",
        "These transitions are descriptive diagnostics on inspected data. "
        "They do not establish a population-level mechanism or independent "
        "generalization.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-a-subset", type=Path, required=True)
    parser.add_argument("--fold-a-baseline", type=Path, required=True)
    parser.add_argument("--fold-a-candidate", type=Path, required=True)
    parser.add_argument("--fold-b-subset", type=Path, required=True)
    parser.add_argument("--fold-b-baseline", type=Path, required=True)
    parser.add_argument("--fold-b-candidate", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.out, args.report):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    result = build_audit(
        fold_a_subset=args.fold_a_subset.resolve(),
        fold_a_baseline=args.fold_a_baseline.resolve(),
        fold_a_candidate=args.fold_a_candidate.resolve(),
        fold_b_subset=args.fold_b_subset.resolve(),
        fold_b_baseline=args.fold_b_baseline.resolve(),
        fold_b_candidate=args.fold_b_candidate.resolve(),
        answers_path=args.answers.resolve(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(render_report(result), encoding="utf-8")
    print(json.dumps({
        name: fold["counts"]["outcomes"]
        for name, fold in result["folds"].items()
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
