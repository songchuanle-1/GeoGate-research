"""Paired scoring and scene bootstrap for the MMScan 3RScan confirmation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import re
import shutil
import string


def load_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def clean_answer(value: str) -> str:
    """Match the pinned LLaVA-3D scanqa_evaluator clean_answer."""
    text = str(value).lower()
    text = re.sub(r"[ ]+$", "", text)
    text = re.sub(r"^[ ]+", "", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\.[ ]{2,}", ". ", text)
    text = re.sub(r"[^a-zA-Z0-9,'\s\-:]+", "", text)
    text = re.sub("\u00e7", "c", text)
    text = re.sub("\u2019", "'", text)
    text = re.sub(r"\bletf\b", "left", text)
    text = re.sub(r"\blet\b", "left", text)
    text = re.sub(r"\btehre\b", "there", text)
    text = re.sub(r"\brigth\b", "right", text)
    text = re.sub(r"\brght\b", "right", text)
    text = re.sub(r"\bbehine\b", "behind", text)
    text = re.sub(r"\btv\b", "TV", text)
    text = re.sub(r"\bchai\b", "chair", text)
    text = re.sub(r"\bwasing\b", "washing", text)
    text = re.sub(r"\bwaslked\b", "walked", text)
    text = re.sub(r"\boclock\b", "o'clock", text)
    text = re.sub(r"\bo'[ ]+clock\b", "o'clock", text)
    text = re.sub(r"\bnone\b", "zero", text)
    for digit, word in enumerate(
        [
            "zero", "one", "two", "three", "four", "five",
            "six", "seven", "eight", "nine", "ten", "eleven",
            "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
            "seventeen", "eighteen", "nineteen", "twenty",
        ]
    ):
        text = re.sub(rf"\b{digit}\b", word, text)
    text = re.sub(r"\b23\b", "twenty-three", text)
    text = re.sub(r"\b([a-zA-Z]+)([0-9])\b", r"\g<1>", text)
    text = re.sub(r"\ba\b ([a-zA-Z]+)", r"\g<1>", text)
    text = re.sub(r"\ban\b ([a-zA-Z]+)", r"\g<1>", text)
    text = re.sub(r"\bthe\b ([a-zA-Z]+)", r"\g<1>", text)
    text = re.sub(r"\bbackwards\b", "backward", text)
    return text


def answer_match(prediction: str, references: list[str]) -> tuple[int, int]:
    raw_match = int(prediction in references)
    if raw_match:
        return 1, 1
    compact_prediction = "".join(prediction.split())
    for reference in references:
        compact_reference = "".join(reference.split())
        if compact_prediction in compact_reference or \
                compact_reference in compact_prediction:
            return raw_match, 1
    return raw_match, 0


def normalize_answer(value: str) -> str:
    text = str(value).lower()
    text = "".join(
        character for character in text
        if character not in set(string.punctuation)
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def simple_tokenize(value: str) -> str:
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def resolve_java_binary() -> Path | None:
    """Find Java without relying on the scheduler's default PATH."""
    candidates = []
    explicit = os.environ.get("GEOGATE_JAVA_BIN")
    if explicit:
        candidates.append(Path(explicit))
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        candidates.append(Path(java_home) / "bin" / "java")
    discovered = shutil.which("java")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def choose_tokenizer() -> tuple[str, object | None]:
    """Use the official PTBTokenizer only when its Java runtime is present."""
    java_binary = resolve_java_binary()
    if java_binary is None:
        return "simple_whitespace_fallback", None
    java_bin_dir = str(java_binary.parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if java_bin_dir not in path_entries:
        os.environ["PATH"] = os.pathsep.join(
            [java_bin_dir, *path_entries]
        )
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    return "ptb_tokenizer", PTBTokenizer()


def tokenize_coco(
    values: dict[str, list[str]],
    tokenizer_name: str,
    tokenizer,
) -> dict[str, list[str]]:
    if tokenizer_name == "simple_whitespace_fallback":
        return {
            key: [simple_tokenize(value) for value in items]
            for key, items in values.items()
        }
    coco = {
        key: [{"caption": value} for value in items]
        for key, items in values.items()
    }
    tokenized = tokenizer.tokenize(coco)
    result = {}
    for key in values:
        normalized = []
        for item in tokenized.get(key, []):
            if isinstance(item, dict):
                value = item.get("caption", "")
            else:
                value = item
            normalized.append(" ".join(str(value).split()))
        result[key] = normalized or [""]
    return result


def tokenize_mapping(
    values: dict[str, str],
    tokenizer_name: str,
    tokenizer,
) -> dict[str, list[str]]:
    return tokenize_coco(
        {key: [value] for key, value in values.items()},
        tokenizer_name,
        tokenizer,
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_output_lengths(values: list[str]) -> dict:
    lengths = [len(simple_tokenize(value).split()) for value in values]
    return {
        "mean_words": mean([float(value) for value in lengths]),
        "median_words": percentile([float(value) for value in lengths], 0.5),
        "p95_words": percentile([float(value) for value in lengths], 0.95),
        "empty": sum(length == 0 for length in lengths),
    }


def prediction_text(row: dict) -> str:
    values = row.get("pred", row.get("text"))
    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError("Expected exactly one prediction per row")
        return str(values[0])
    if values is None:
        raise ValueError("Prediction row has no pred/text field")
    return str(values)


def reference_texts(row: dict) -> list[str]:
    values = row.get("gt", row.get("answers", row.get("text")))
    if isinstance(values, list):
        return [str(value) for value in values]
    if values is None:
        raise ValueError("Prediction row has no gt/answers field")
    return [str(values)]


def pair_rows(
    baseline_path: Path,
    candidate_path: Path,
) -> list[dict]:
    baseline = {str(row["question_id"]): row for row in load_jsonl(baseline_path)}
    candidate = {
        str(row["question_id"]): row
        for row in load_jsonl(candidate_path)
    }
    if set(baseline) != set(candidate):
        missing = sorted(set(baseline) - set(candidate))
        extra = sorted(set(candidate) - set(baseline))
        raise ValueError(
            f"Question mismatch: {len(missing)} missing, {len(extra)} extra"
        )
    rows = []
    for question_id in sorted(baseline):
        reference = baseline[question_id]
        treated = candidate[question_id]
        if reference["prompt"] != treated["prompt"]:
            raise ValueError(f"Prompt mismatch for {question_id}")
        if reference_texts(reference) != reference_texts(treated):
            raise ValueError(f"Reference mismatch for {question_id}")
        rows.append({
            "question_id": question_id,
            "prompt": reference["prompt"],
            "references": reference_texts(reference),
            "baseline": prediction_text(reference),
            "candidate": prediction_text(treated),
            "baseline_latency_s": float(reference.get("latency_s", 0.0)),
            "candidate_latency_s": float(treated.get("latency_s", 0.0)),
        })
    return rows


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = round(quantile * (len(ordered) - 1))
    return ordered[min(len(ordered) - 1, max(0, index))]


def scene_bootstrap(
    em_deltas: dict[str, float],
    refined_em_deltas: dict[str, float],
    cider_deltas: dict[str, float],
    scenes: dict[str, str],
    iterations: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    question_ids = sorted(em_deltas)
    if set(question_ids) != set(cider_deltas) or \
            set(question_ids) != set(refined_em_deltas) or \
            set(question_ids) != set(scenes):
        raise ValueError("Metrics and scenes must cover the same question IDs")
    grouped = defaultdict(list)
    for question_id in question_ids:
        grouped[scenes[question_id]].append(question_id)
    scene_ids = sorted(grouped)
    rng = random.Random(seed)
    em_estimates = []
    refined_em_estimates = []
    cider_estimates = []
    for _ in range(iterations):
        sampled = [scene_ids[rng.randrange(len(scene_ids))]
                   for _ in scene_ids]
        keys = [
            question_id
            for scene in sampled
            for question_id in grouped[scene]
        ]
        em_estimates.append(
            sum(em_deltas[key] for key in keys) / len(keys)
        )
        refined_em_estimates.append(
            sum(refined_em_deltas[key] for key in keys) / len(keys)
        )
        cider_estimates.append(
            sum(cider_deltas[key] for key in keys) / len(keys)
        )
    return {
        "em": (
            percentile(em_estimates, 0.025),
            percentile(em_estimates, 0.975),
        ),
        "refined_em": (
            percentile(refined_em_estimates, 0.025),
            percentile(refined_em_estimates, 0.975),
        ),
        "cider": (
            percentile(cider_estimates, 0.025),
            percentile(cider_estimates, 0.975),
        ),
    }


def score(
    baseline_path: Path,
    candidate_path: Path,
    question_path: Path,
    iterations: int = 10_000,
    seed: int = 20260917,
) -> dict:
    from pycocoevalcap.cider.cider import Cider

    rows = pair_rows(baseline_path, candidate_path)
    questions = {
        str(row["prompt_id"]): row
        for row in json.loads(Path(question_path).read_text(encoding="utf-8"))
    }
    if set(questions) != {row["question_id"] for row in rows}:
        raise ValueError("Question metadata does not match prediction IDs")

    em_deltas = {}
    refined_em_deltas = {}
    baseline_em = 0
    candidate_em = 0
    baseline_refined_em = 0
    candidate_refined_em = 0
    references = {}
    baseline_hypotheses = {}
    candidate_hypotheses = {}
    subclasses = {}
    baseline_raw_outputs = []
    candidate_raw_outputs = []
    for row in rows:
        question_id = row["question_id"]
        clean_references = [
            clean_answer(value) for value in row["references"]
        ]
        baseline_clean = clean_answer(row["baseline"])
        candidate_clean = clean_answer(row["candidate"])
        baseline_match = answer_match(baseline_clean, clean_references)
        candidate_match = answer_match(candidate_clean, clean_references)
        baseline_em += baseline_match[0]
        candidate_em += candidate_match[0]
        baseline_refined_em += baseline_match[1]
        candidate_refined_em += candidate_match[1]
        em_deltas[question_id] = candidate_match[0] - baseline_match[0]
        refined_em_deltas[question_id] = (
            candidate_match[1] - baseline_match[1]
        )
        references[question_id] = clean_references
        baseline_hypotheses[question_id] = [baseline_clean]
        candidate_hypotheses[question_id] = [candidate_clean]
        subclasses[question_id] = str(
            questions[question_id].get("sub_class", "unknown")
        )
        baseline_raw_outputs.append(row["baseline"])
        candidate_raw_outputs.append(row["candidate"])

    tokenizer_name, tokenizer = choose_tokenizer()
    tokenized_references = tokenize_coco(
        references,
        tokenizer_name,
        tokenizer,
    )
    tokenized_baseline = tokenize_mapping(
        {
            key: values[0]
            for key, values in baseline_hypotheses.items()
        },
        tokenizer_name,
        tokenizer,
    )
    tokenized_candidate = tokenize_mapping(
        {
            key: values[0]
            for key, values in candidate_hypotheses.items()
        },
        tokenizer_name,
        tokenizer,
    )
    baseline_cider, baseline_scores = Cider().compute_score(
        tokenized_references,
        tokenized_baseline,
    )
    candidate_cider, candidate_scores = Cider().compute_score(
        tokenized_references,
        tokenized_candidate,
    )
    keys = sorted(references)
    cider_deltas = {
        key: float(candidate_scores[index] - baseline_scores[index])
        for index, key in enumerate(keys)
    }
    scenes = {
        row["question_id"]: str(questions[row["question_id"]]["video"])
        for row in rows
    }
    intervals = scene_bootstrap(
        em_deltas,
        refined_em_deltas,
        cider_deltas,
        scenes,
        iterations,
        seed,
    )
    n = len(rows)
    em_delta = (candidate_em - baseline_em) / n
    refined_em_delta = (
        candidate_refined_em - baseline_refined_em
    ) / n
    cider_delta = float(candidate_cider - baseline_cider)
    baseline_latencies = [row["baseline_latency_s"] for row in rows]
    candidate_latencies = [row["candidate_latency_s"] for row in rows]
    cider_delta_by_id = cider_deltas
    subclass_metrics = {}
    for subclass in sorted(set(subclasses.values())):
        ids = [
            question_id for question_id, value in subclasses.items()
            if value == subclass
        ]
        subclass_metrics[subclass] = {
            "questions": len(ids),
            "scenes": len({scenes[question_id] for question_id in ids}),
            "delta_em": mean([em_deltas[question_id] for question_id in ids]),
            "delta_refined_em": mean([
                refined_em_deltas[question_id] for question_id in ids
            ]),
            "delta_cider": mean([
                cider_delta_by_id[question_id] for question_id in ids
            ]),
        }
    return {
        "schema": "geogate-mmscan-3rscan-score-v2",
        "counts": {
            "questions": n,
            "scenes": len(set(scenes.values())),
        },
        "metric_protocol": {
            "source_commit": "91d979dca9998174c8eda50744de8da448e0ba0c",
            "source_path": "llava/eval/scanqa_evaluator.py",
            "clean_answer": "exact clean_answer semantics from the pinned evaluator",
            "em": "clean_answer + exact membership",
            "refined_em": "clean_answer + exact substring containment",
            "cider_tokenizer": tokenizer_name,
            "official_cider_equivalent": tokenizer_name == "ptb_tokenizer",
            "java_available": resolve_java_binary() is not None,
        },
        "metrics": {
            "baseline_em": baseline_em / n,
            "candidate_em": candidate_em / n,
            "delta_em": em_delta,
            "baseline_refined_em": baseline_refined_em / n,
            "candidate_refined_em": candidate_refined_em / n,
            "delta_refined_em": refined_em_delta,
            "baseline_cider": float(baseline_cider),
            "candidate_cider": float(candidate_cider),
            "delta_cider": cider_delta,
        },
        "scene_bootstrap": {
            "iterations": iterations,
            "seed": seed,
            "em_95ci": list(intervals["em"]),
            "refined_em_95ci": list(intervals["refined_em"]),
            "cider_95ci": list(intervals["cider"]),
        },
        "subclass_metrics": subclass_metrics,
        "output_lengths": {
            "baseline": summarize_output_lengths(baseline_raw_outputs),
            "candidate": summarize_output_lengths(candidate_raw_outputs),
        },
        "latency": {
            "baseline_median_s": percentile(baseline_latencies, 0.5),
            "candidate_median_s": percentile(candidate_latencies, 0.5),
            "baseline_p95_s": percentile(baseline_latencies, 0.95),
            "candidate_p95_s": percentile(candidate_latencies, 0.95),
        },
        "promotion_gate": {
            "primary_endpoint": "EM",
            "passes": bool(
                em_delta > 0
                and intervals["em"][0] > 0
                and n == 4_934
                and len(set(scenes.values())) == 234
            ),
            "paper_updated": False,
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--baseline", type=Path, required=True)
    result.add_argument("--candidate", type=Path, required=True)
    result.add_argument("--questions", type=Path, required=True)
    result.add_argument("--iterations", type=int, default=10_000)
    result.add_argument("--seed", type=int, default=20260917)
    result.add_argument("--out", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite score output")
    result = score(
        args.baseline,
        args.candidate,
        args.questions,
        args.iterations,
        args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
