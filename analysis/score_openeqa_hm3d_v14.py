#!/usr/bin/env python3
"""Score frozen v14 paired official grades with episode bootstrap."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re


FROZEN_SEED = 20260913
FROZEN_ITERATIONS = 10000
NONINFERIORITY_MARGIN = -2.0
CONDITIONS = ("native", "geogate_L14_R010")


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def percentile(values, probability):
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, round(probability * (len(ordered) - 1))),
    )
    return ordered[index]


def episode_interval(deltas, episodes):
    grouped = defaultdict(list)
    for delta, episode in zip(deltas, episodes):
        grouped[episode].append(float(delta))
    labels = sorted(grouped)
    rng = random.Random(FROZEN_SEED)
    estimates = []
    for _ in range(FROZEN_ITERATIONS):
        sampled = [
            labels[rng.randrange(len(labels))]
            for _ in labels
        ]
        values = [
            value
            for episode in sampled
            for value in grouped[episode]
        ]
        estimates.append(sum(values) / len(values))
    return {
        "mean": sum(deltas) / len(deltas),
        "ci": [
            percentile(estimates, 0.025),
            percentile(estimates, 0.975),
        ],
        "episodes": len(labels),
    }


def normalized_tokens(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def exact_and_f1(prediction, references):
    prediction_tokens = normalized_tokens(prediction)
    best_exact = 0
    best_f1 = 0.0
    for reference in references:
        reference_tokens = normalized_tokens(reference)
        best_exact = max(
            best_exact,
            int(prediction_tokens == reference_tokens),
        )
        if not prediction_tokens and not reference_tokens:
            f1 = 1.0
        elif not prediction_tokens or not reference_tokens:
            f1 = 0.0
        else:
            prediction_count = Counter(prediction_tokens)
            reference_count = Counter(reference_tokens)
            overlap = sum(
                (prediction_count & reference_count).values()
            )
            precision = overlap / len(prediction_tokens)
            recall = overlap / len(reference_tokens)
            f1 = (
                0.0 if precision + recall == 0
                else 2 * precision * recall / (precision + recall)
            )
        best_f1 = max(best_f1, f1)
    return best_exact, best_f1


def load_grades(path):
    grades = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(grades, dict):
        raise ValueError("Official grade file must be a question-id dict")
    result = {}
    for question_id, grade in grades.items():
        if (
            isinstance(grade, bool)
            or not isinstance(grade, int)
            or not 1 <= grade <= 5
        ):
            raise ValueError(
                f"Invalid official grade for {question_id}: {grade}"
            )
        result[str(question_id)] = grade
    return result


def load_pairs(raw_path):
    rows = [
        json.loads(line)
        for line in Path(raw_path).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    pairs = defaultdict(dict)
    order = []
    for row in rows:
        question_id = str(row["question_id"])
        if question_id not in pairs:
            order.append(question_id)
        condition = row["condition"]
        if condition in pairs[question_id]:
            raise ValueError(
                f"Duplicate {question_id}/{condition}"
            )
        pairs[question_id][condition] = row
    for question_id in order:
        if set(pairs[question_id]) != set(CONDITIONS):
            raise ValueError(f"Incomplete pair: {question_id}")
    return pairs, order


def summarize(
    raw_path,
    dataset_path,
    audit_path,
    native_grades_path,
    geogate_grades_path,
):
    audit = json.loads(
        Path(audit_path).read_text(encoding="utf-8")
    )
    if (
        audit.get("state") != "passed"
        or not audit.get("scoring_authorized")
    ):
        raise ValueError(
            "Structural audit has not authorized official scoring"
        )
    if audit.get("raw", {}).get("sha256") != digest_file(raw_path):
        raise ValueError("Raw file does not match structural audit")

    pairs, question_ids = load_pairs(raw_path)
    dataset = {
        str(row["question_id"]): row
        for row in json.loads(
            Path(dataset_path).read_text(encoding="utf-8")
        )
        if str(row.get("episode_history", "")).startswith("hm3d-v0/")
    }
    native_grades = load_grades(native_grades_path)
    geogate_grades = load_grades(geogate_grades_path)
    expected = set(question_ids)
    if (
        set(dataset) != expected
        or set(native_grades) != expected
        or set(geogate_grades) != expected
    ):
        raise ValueError(
            "Dataset, raw pairs, and grade question IDs differ"
        )

    native_scores = []
    geogate_scores = []
    deltas = []
    episodes = []
    categories = []
    lexical = {
        condition: {"exact": [], "token_f1": []}
        for condition in CONDITIONS
    }
    for question_id in question_ids:
        item = dataset[question_id]
        episode = str(item["episode_history"])
        category = str(item.get("category", "unknown"))
        native_score = 25.0 * (
            native_grades[question_id] - 1
        )
        geogate_score = 25.0 * (
            geogate_grades[question_id] - 1
        )
        native_scores.append(native_score)
        geogate_scores.append(geogate_score)
        deltas.append(geogate_score - native_score)
        episodes.append(episode)
        categories.append(category)
        references = [item["answer"]] + list(
            item.get("extra_answers", [])
        )
        for condition in CONDITIONS:
            exact, token_f1 = exact_and_f1(
                pairs[question_id][condition]["answer"],
                references,
            )
            lexical[condition]["exact"].append(exact)
            lexical[condition]["token_f1"].append(token_f1)

    primary = episode_interval(deltas, episodes)
    category_results = {}
    for category in sorted(set(categories)):
        indices = [
            index
            for index, value in enumerate(categories)
            if value == category
        ]
        category_results[category] = episode_interval(
            [deltas[index] for index in indices],
            [episodes[index] for index in indices],
        )
        category_results[category]["questions"] = len(indices)

    lower = primary["ci"][0]
    if lower > 0:
        decision = "supports_directional_improvement"
    elif lower >= NONINFERIORITY_MARGIN:
        decision = "supports_frozen_noninferior_transfer"
    else:
        decision = "fails_frozen_noninferiority_confirmation"
    return {
        "state": "passed",
        "decision": decision,
        "questions": len(question_ids),
        "episodes": len(set(episodes)),
        "official_llm_match": {
            "native_mean": sum(native_scores) / len(native_scores),
            "geogate_mean": (
                sum(geogate_scores) / len(geogate_scores)
            ),
            "paired_delta": primary,
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "wins": sum(delta > 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "losses": sum(delta < 0 for delta in deltas),
        },
        "lexical_diagnostics": {
            condition: {
                metric: sum(values) / len(values)
                for metric, values in metrics.items()
            }
            for condition, metrics in lexical.items()
        },
        "category_deltas": category_results,
        "bootstrap": {
            "unit": "episode",
            "iterations": FROZEN_ITERATIONS,
            "seed": FROZEN_SEED,
            "interval": 0.95,
        },
        "hashes": {
            "raw": digest_file(raw_path),
            "dataset": digest_file(dataset_path),
            "structural_audit": digest_file(audit_path),
            "native_grades": digest_file(native_grades_path),
            "geogate_grades": digest_file(geogate_grades_path),
            "scorer": digest_file(__file__),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--native-grades", type=Path, required=True)
    parser.add_argument("--geogate-grades", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    result = summarize(
        args.raw,
        args.dataset,
        args.audit,
        args.native_grades,
        args.geogate_grades,
    )
    if not all(
        math.isfinite(value)
        for value in result["official_llm_match"][
            "paired_delta"
        ]["ci"]
    ):
        raise ValueError("Non-finite primary confidence interval")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
