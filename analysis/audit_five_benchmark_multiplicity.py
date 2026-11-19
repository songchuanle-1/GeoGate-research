#!/usr/bin/env python3
"""CPU-only family-wise sensitivity audit for the five 3D-LLaVA endpoints.

The audit reuses frozen prediction files. It does not rerun a model, select a
new configuration, or turn the marginal benchmark intervals into independent
confirmation. Its purpose is narrower: apply one prespecified Holm correction
to the five benchmark-level paired effects already reported in the paper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Callable, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FROZEN_ITERATIONS = 10000
FROZEN_SEED = 20260918
FAMILY_ALPHA = 0.05
DEFAULT_OUTPUT = (
    ROOT
    / "results/raw/followup/five_benchmark_multiplicity_20260918.json"
)
DEFAULT_MARKDOWN = (
    ROOT / "results/FIVE_BENCHMARK_MULTIPLICITY_20260918.md"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_or_jsonl(path: Path):
    raw = path.read_text(encoding="utf-8").strip()
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(rows, dict):
        rows = rows.get("answers", rows.get("data", [rows]))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list/object or JSONL rows: {path}")
    return rows


def unique_by(rows: Iterable[dict], fields: tuple[str, ...], label: str) -> dict:
    keyed = {}
    for row in rows:
        key = tuple(str(row[field]) for field in fields)
        if key in keyed:
            raise ValueError(f"Duplicate key in {label}: {key}")
        keyed[key] = row
    return keyed


def require_identical_keys(reference: dict, candidate: dict, label: str) -> list:
    if set(reference) != set(candidate):
        missing = sorted(set(reference) - set(candidate))
        extra = sorted(set(candidate) - set(reference))
        raise ValueError(
            f"{label} requires identical keys; missing={missing[:5]}, "
            f"extra={extra[:5]}"
        )
    return sorted(reference)


def require_same_prompt(reference: dict, candidate: dict, label: str) -> None:
    for key in reference:
        if reference[key].get("prompt") != candidate[key].get("prompt"):
            raise ValueError(f"{label} prompt mismatch: {key}")


def scene_cluster_bootstrap(
    deltas: np.ndarray,
    scenes: list[str],
    iterations: int,
    seed: int,
) -> tuple[list[float], float, float]:
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.ndim != 1 or len(deltas) == 0:
        raise ValueError("Deltas must be a nonempty one-dimensional array")
    if len(scenes) != len(deltas):
        raise ValueError("Each delta must have one scene label")
    if not np.all(np.isfinite(deltas)):
        raise ValueError("Deltas must be finite")

    grouped_values: dict[str, list[float]] = defaultdict(list)
    for value, scene in zip(deltas.tolist(), scenes):
        grouped_values[str(scene)].append(value)
    scene_ids = sorted(grouped_values)
    if len(scene_ids) < 2:
        raise ValueError("At least two scene clusters are required")

    sums = np.asarray(
        [sum(grouped_values[scene]) for scene in scene_ids],
        dtype=np.float64,
    )
    counts = np.asarray(
        [len(grouped_values[scene]) for scene in scene_ids],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0,
        len(scene_ids),
        size=(iterations, len(scene_ids)),
    )
    estimates = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975]).tolist()

    observed = float(deltas.mean())
    centered_null = estimates - observed
    extreme = np.count_nonzero(np.abs(centered_null) >= abs(observed))
    # Add-one smoothing keeps the p-value finite for a fully separated effect.
    p_value = float((extreme + 1) / (iterations + 1))
    return [float(lower), float(upper)], observed, p_value


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    if not p_values:
        raise ValueError("Holm correction requires at least one p-value")
    if any(not math.isfinite(value) or value < 0 or value > 1
           for value in p_values.values()):
        raise ValueError("p-values must be finite and in [0, 1]")

    ordered = sorted(p_values, key=lambda name: (p_values[name], name))
    family_size = len(ordered)
    adjusted = {}
    running = 0.0
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (family_size - rank) * p_values[name])
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def compact_endpoint(
    *,
    benchmark: str,
    metric: str,
    comparison: str,
    deltas: np.ndarray,
    scenes: list[str],
    reference_values: np.ndarray,
    candidate_values: np.ndarray,
    iterations: int,
    seed: int,
) -> dict:
    reference_values = np.asarray(reference_values, dtype=np.float64) * 100.0
    candidate_values = np.asarray(candidate_values, dtype=np.float64) * 100.0
    deltas = np.asarray(deltas, dtype=np.float64) * 100.0
    ci, delta, p_value = scene_cluster_bootstrap(
        deltas,
        scenes,
        iterations,
        seed,
    )
    return {
        "benchmark": benchmark,
        "metric": metric,
        "comparison": comparison,
        "unit": "percentage_points",
        "n_items": int(len(deltas)),
        "n_scenes": int(len(set(scenes))),
        "reference": float(reference_values.mean()),
        "candidate": float(candidate_values.mean()),
        "delta": float(delta),
        "marginal_scene_bootstrap_95ci": ci,
        "bootstrap_two_sided_p": p_value,
        "bootstrap_seed": seed,
    }


def build_qa_endpoint(
    *,
    benchmark: str,
    reference_path: Path,
    candidate_path: Path,
    gt_path: Path,
    scene_from_row: Callable[[dict], str],
    iterations: int,
    seed: int,
) -> tuple[dict, list[Path]]:
    from tools.scanqa_score import answer_match, clean_answer

    reference = unique_by(
        load_json_or_jsonl(reference_path),
        ("question_id",),
        str(reference_path),
    )
    candidate = unique_by(
        load_json_or_jsonl(candidate_path),
        ("question_id",),
        str(candidate_path),
    )
    keys = require_identical_keys(reference, candidate, benchmark)
    require_same_prompt(reference, candidate, benchmark)
    gt = unique_by(
        load_json_or_jsonl(gt_path),
        ("question_id",),
        str(gt_path),
    )
    if set(gt) != set(reference):
        raise ValueError(f"{benchmark} ground truth does not match predictions")

    deltas = []
    reference_values = []
    candidate_values = []
    scenes = []
    for key in keys:
        references = [
            clean_answer(text)
            for text in gt[key]["text"]
        ]
        reference_value = answer_match(
            clean_answer(reference[key]["text"]),
            references,
        )[0]
        candidate_value = answer_match(
            clean_answer(candidate[key]["text"]),
            references,
        )[0]
        reference_values.append(reference_value)
        candidate_values.append(candidate_value)
        deltas.append(candidate_value - reference_value)
        scenes.append(str(scene_from_row(gt[key])))

    endpoint = compact_endpoint(
        benchmark=benchmark,
        metric="EM",
        comparison="GeoGate_R10_minus_baseline",
        deltas=np.asarray(deltas),
        scenes=scenes,
        reference_values=np.asarray(reference_values),
        candidate_values=np.asarray(candidate_values),
        iterations=iterations,
        seed=seed,
    )
    return endpoint, [reference_path, candidate_path, gt_path]


def build_referring_endpoint(
    *,
    benchmark: str,
    reference_path: Path,
    candidate_path: Path,
    iterations: int,
    seed: int,
) -> tuple[dict, list[Path]]:
    reference = unique_by(
        load_json_or_jsonl(reference_path),
        ("scene_id", "question_id"),
        str(reference_path),
    )
    candidate = unique_by(
        load_json_or_jsonl(candidate_path),
        ("scene_id", "question_id"),
        str(candidate_path),
    )
    keys = require_identical_keys(reference, candidate, benchmark)
    require_same_prompt(reference, candidate, benchmark)

    reference_values = np.asarray(
        [float(reference[key]["iou"]) for key in keys],
        dtype=np.float64,
    )
    candidate_values = np.asarray(
        [float(candidate[key]["iou"]) for key in keys],
        dtype=np.float64,
    )
    if np.any((reference_values < 0) | (reference_values > 1)):
        raise ValueError(f"{benchmark} baseline IoU outside [0, 1]")
    if np.any((candidate_values < 0) | (candidate_values > 1)):
        raise ValueError(f"{benchmark} candidate IoU outside [0, 1]")

    endpoint = compact_endpoint(
        benchmark=benchmark,
        metric="mIoU",
        comparison="GeoGate_R10_minus_baseline",
        deltas=candidate_values - reference_values,
        scenes=[key[0] for key in keys],
        reference_values=reference_values,
        candidate_values=candidate_values,
        iterations=iterations,
        seed=seed,
    )
    return endpoint, [reference_path, candidate_path]


def build_scan2cap_endpoint(
    *,
    reference_path: Path,
    candidate_path: Path,
    pred_attrs_path: Path,
    gt_attrs_path: Path,
    corpus_path: Path,
    eval_dir: Path,
    iterations: int,
    seed: int,
    iou_threshold: float,
) -> tuple[dict, list[Path]]:
    # Heavy optional dependencies are imported only for the official Scan2Cap
    # endpoint, so pure multiplicity tests can run in a lightweight environment.
    import torch
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    eval_dir = eval_dir.resolve()
    sys.path.insert(0, str(eval_dir))
    import box_utils

    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    if not isinstance(corpus, dict) or not corpus:
        raise ValueError("Scan2Cap corpus must be a nonempty mapping")
    corpus_keys = sorted(corpus)
    if len(corpus_keys) != len(set(corpus_keys)):
        raise ValueError("Scan2Cap corpus contains duplicate keys")

    instance_attrs = torch.load(
        pred_attrs_path,
        map_location="cpu",
        weights_only=False,
    )
    gt_attrs = torch.load(
        gt_attrs_path,
        map_location="cpu",
        weights_only=False,
    )

    def prediction_map(path: Path) -> dict[str, str]:
        rows = load_json_or_jsonl(path)
        keyed = {}
        for row in rows:
            key = f"{row['scene_id']}|{row['gt_id']}"
            if key in keyed:
                raise ValueError(f"Duplicate Scan2Cap key in {path}: {key}")
            scene_id = str(row["scene_id"])
            pred_id = int(row["pred_id"])
            gt_id = int(row["gt_id"])
            pred_locs = instance_attrs[scene_id]["locs"][pred_id].tolist()
            gt_locs = gt_attrs[scene_id]["locs"][gt_id].tolist()
            pred_corners = box_utils.construct_bbox_corners(
                pred_locs[:3],
                pred_locs[3:],
            )
            gt_corners = box_utils.construct_bbox_corners(
                gt_locs[:3],
                gt_locs[3:],
            )
            iou = float(box_utils.box3d_iou(pred_corners, gt_corners))
            text = str(row["text"]) if iou >= iou_threshold else ""
            keyed[key] = f"sos {text} eos".replace("\n", " ")
        extra = sorted(set(keyed) - set(corpus_keys))
        if extra:
            raise ValueError(
                "Scan2Cap predictions contain keys outside the corpus; "
                f"extra={extra[:5]}"
            )
        return keyed

    reference_text = prediction_map(reference_path)
    candidate_text = prediction_map(candidate_path)
    missing_reference = sorted(set(corpus_keys) - set(reference_text))
    missing_candidate = sorted(set(corpus_keys) - set(candidate_text))
    if missing_reference or missing_candidate:
        if missing_reference != missing_candidate:
            raise ValueError("Scan2Cap baseline/candidate coverage differs")
    targets = {
        key: [{"caption": caption} for caption in corpus[key]]
        for key in corpus_keys
    }
    reference = {
        key: [{"caption": reference_text.get(key, "sos eos")}]
        for key in corpus_keys
    }
    candidate = {
        key: [{"caption": candidate_text.get(key, "sos eos")}]
        for key in corpus_keys
    }

    tokenizer = PTBTokenizer()
    targets = tokenizer.tokenize(targets)
    reference = tokenizer.tokenize(reference)
    candidate = tokenizer.tokenize(candidate)

    reference_score, reference_scores = Cider().compute_score(
        targets,
        reference,
    )
    candidate_score, candidate_scores = Cider().compute_score(
        targets,
        candidate,
    )
    reference_scores = np.asarray(reference_scores, dtype=np.float64)
    candidate_scores = np.asarray(candidate_scores, dtype=np.float64)
    if reference_scores.shape != (len(corpus_keys),):
        raise ValueError("Unexpected Scan2Cap baseline CIDEr shape")
    if candidate_scores.shape != (len(corpus_keys),):
        raise ValueError("Unexpected Scan2Cap candidate CIDEr shape")

    endpoint = compact_endpoint(
        benchmark="Scan2Cap",
        metric="CIDEr@0.25",
        comparison="GeoGate_R10_minus_baseline",
        deltas=candidate_scores - reference_scores,
        scenes=[key.split("|", 1)[0] for key in corpus_keys],
        reference_values=reference_scores,
        candidate_values=candidate_scores,
        iterations=iterations,
        seed=seed,
    )
    endpoint["official_score"] = {
        "reference": float(reference_score) * 100.0,
        "candidate": float(candidate_score) * 100.0,
        "delta": float(candidate_score - reference_score) * 100.0,
    }
    endpoint["scan2cap_missing_prediction_count"] = len(missing_reference)
    endpoint["scan2cap_prediction_count"] = len(reference_text)
    return endpoint, [
        reference_path,
        candidate_path,
        pred_attrs_path,
        gt_attrs_path,
        corpus_path,
        eval_dir / "box_utils.py",
    ]


def build_audit(
    *,
    root: Path = ROOT,
    input_root: Path | None = None,
    iterations: int = FROZEN_ITERATIONS,
    seed: int = FROZEN_SEED,
) -> dict:
    if iterations < 1000:
        raise ValueError("At least 1,000 bootstrap iterations are required")
    if input_root is None:
        input_root = root
    input_root = input_root.resolve()
    raw = input_root / "results/raw"
    scan2cap_root = (
        input_root
        / "third_party/3D-LLaVA/playground/data/eval_info/densecap_scanrefer"
    )
    scanqa_gt = (
        input_root
        / "third_party/LLaVA-3D/playground/data/annotations/"
        "llava3d_scanqa_val_answer.json"
    )
    sqa3d_gt = input_root / "tmp/sqa3d_test_answer.jsonl"

    endpoints = {}
    source_paths = []
    endpoint, paths = build_qa_endpoint(
        benchmark="3D-LLaVA ScanQA",
        reference_path=raw / "reviewer/3dl_full_baseline.jsonl",
        candidate_path=raw / "reviewer/3dl_full_geogate.jsonl",
        gt_path=scanqa_gt,
        scene_from_row=lambda row: row["question_id"].split("-")[1],
        iterations=iterations,
        seed=seed,
    )
    endpoints["scanqa"] = endpoint
    source_paths.extend(paths)

    endpoint, paths = build_qa_endpoint(
        benchmark="3D-LLaVA SQA3D",
        reference_path=raw / "3dl_sqa3d_baseline.jsonl",
        candidate_path=raw / "3dl_sqa3d_geogate.jsonl",
        gt_path=sqa3d_gt,
        scene_from_row=lambda row: str(row["scene_id"]),
        iterations=iterations,
        seed=seed + 1,
    )
    endpoints["sqa3d"] = endpoint
    source_paths.extend(paths)

    endpoint, paths = build_referring_endpoint(
        benchmark="3D-LLaVA ScanRefer",
        reference_path=raw / "scanrefer_baseline.jsonl",
        candidate_path=raw / "scanrefer_geogate_L14_R010.jsonl",
        iterations=iterations,
        seed=seed + 2,
    )
    endpoints["scanrefer"] = endpoint
    source_paths.extend(paths)

    endpoint, paths = build_referring_endpoint(
        benchmark="3D-LLaVA Multi3DRefer",
        reference_path=(
            raw / "followup/review_v3_20260908/"
            "multi3drefer_baseline.jsonl"
        ),
        candidate_path=(
            raw / "followup/review_v3_20260908/"
            "multi3drefer_geogate_L14_R010.jsonl"
        ),
        iterations=iterations,
        seed=seed + 3,
    )
    endpoints["multi3drefer"] = endpoint
    source_paths.extend(paths)

    endpoint, paths = build_scan2cap_endpoint(
        reference_path=raw / "scan2cap_baseline.jsonl",
        candidate_path=raw / "scan2cap_geogate_L14_R010.jsonl",
        pred_attrs_path=scan2cap_root / "scannet_mask3d_val_attributes.pt",
        gt_attrs_path=scan2cap_root / "scannet_val_attributes.pt",
        corpus_path=scan2cap_root / "scan2cap_val_corpus.json",
        eval_dir=(
            input_root / "third_party/3D-LLaVA/llava/eval"
        ),
        iterations=iterations,
        seed=seed + 4,
        iou_threshold=0.25,
    )
    endpoints["scan2cap"] = endpoint
    source_paths.extend(paths)

    raw_p_values = {
        name: endpoint["bootstrap_two_sided_p"]
        for name, endpoint in endpoints.items()
    }
    adjusted = holm_adjust(raw_p_values)
    for name, endpoint in endpoints.items():
        endpoint["holm_adjusted_p"] = float(adjusted[name])
        endpoint["holm_reject_at_0.05"] = bool(adjusted[name] < FAMILY_ALPHA)
        endpoint["marginal_interval_includes_zero"] = bool(
            endpoint["marginal_scene_bootstrap_95ci"][0] <= 0.0
            <= endpoint["marginal_scene_bootstrap_95ci"][1]
        )

    expected_counts = {
        "scanqa": 4675,
        "sqa3d": 3519,
        "scan2cap": 2068,
        "scanrefer": 9508,
        "multi3drefer": 11120,
    }
    for name, expected in expected_counts.items():
        if endpoints[name]["n_items"] != expected:
            raise ValueError(
                f"{name} has {endpoints[name]['n_items']} items; "
                f"expected {expected}"
            )

    source_manifest = {}
    for path in sorted(set(source_paths), key=lambda item: str(item)):
        source_manifest[str(path.relative_to(root))] = {
            "sha256": sha256(path),
            "size": path.stat().st_size,
        }

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "post_hoc_multiplicity_sensitivity_audit",
        "scope": {
            "family": [
                "3D-LLaVA ScanQA EM",
                "3D-LLaVA SQA3D EM",
                "3D-LLaVA Scan2Cap CIDEr@0.25",
                "3D-LLaVA ScanRefer mIoU",
                "3D-LLaVA Multi3DRefer mIoU",
            ],
            "family_size": 5,
            "multiplicity_correction": "Holm step-down",
            "alpha": FAMILY_ALPHA,
            "bootstrap_unit": "scene",
            "paired": True,
            "iterations": iterations,
            "base_seed": seed,
            "model_rerun": False,
            "independent_confirmation": False,
            "selection_adjusted": False,
            "input_root": (
                str(input_root.relative_to(root))
                if input_root.is_relative_to(root)
                else str(input_root)
            ),
            "notes": (
                "The five benchmark-level primary endpoints form the tested "
                "family. This post-hoc audit addresses endpoint multiplicity "
                "only; it does not address selection of L14/R10 or the lack "
                "of an untouched confirmatory split."
            ),
        },
        "sources": source_manifest,
        "endpoints": endpoints,
        "decision": {
            "reject_names": [
                name for name, endpoint in endpoints.items()
                if endpoint["holm_reject_at_0.05"]
            ],
            "fail_to_reject_names": [
                name for name, endpoint in endpoints.items()
                if not endpoint["holm_reject_at_0.05"]
            ],
        },
        "generated_by": "analysis/audit_five_benchmark_multiplicity.py",
    }


def render_markdown(audit: dict) -> str:
    scope = audit["scope"]
    lines = [
        "# 五基准多重比较边界审计",
        "",
        f"日期：2026-09-18。状态：`{audit['status']}`，"
        f"family size `{scope['family_size']}`，"
        f"`{scope['multiplicity_correction']}`，"
        f"alpha `{scope['alpha']}`，"
        f"`{scope['iterations']}` 次 paired scene-cluster bootstrap。",
        "",
        "该审计仅复用冻结预测，不重新运行模型，也不做新的配置选择。",
        "它校正的是五个 benchmark 主指标组成的比较族，不解决 L14/R10 "
        "选择问题，也不把被检查过的 split 变成独立确认域。",
        "",
        "| Endpoint | N / scenes | Delta | Marginal 95% CI | Bootstrap p | Holm p |",
        "|---|---:|---:|---|---:|---:|",
    ]
    order = [
        "scanqa",
        "sqa3d",
        "scan2cap",
        "scanrefer",
        "multi3drefer",
    ]
    for name in order:
        endpoint = audit["endpoints"][name]
        low, high = endpoint["marginal_scene_bootstrap_95ci"]
        lines.append(
            f"| {endpoint['benchmark']} {endpoint['metric']} "
            f"| {endpoint['n_items']} / {endpoint['n_scenes']} "
            f"| {endpoint['delta']:+.4f} "
            f"| [{low:+.4f}, {high:+.4f}] "
            f"| {endpoint['bootstrap_two_sided_p']:.4f} "
            f"| {endpoint['holm_adjusted_p']:.4f} |"
        )
    lines.extend([
        "",
        "Holm 校正后拒绝零效应的端点："
        + (", ".join(audit["decision"]["reject_names"]) or "无")
        + "。未能拒绝零效应的端点："
        + (", ".join(audit["decision"]["fail_to_reject_names"]) or "无")
        + "。",
        "",
        "判决只说明在 5% family-wise 错误率下哪些冻结边际差异仍然可见。"
        "未显著不等于等价或非劣；显著也不等于独立确认。原始 JSON 保存了"
        "完整输入哈希、每个端点的随机种子和 bootstrap 设置。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--iterations", type=int, default=FROZEN_ITERATIONS)
    parser.add_argument("--seed", type=int, default=FROZEN_SEED)
    args = parser.parse_args()

    audit = build_audit(
        root=args.root.resolve(),
        input_root=(
            args.input_root.resolve()
            if args.input_root is not None
            else None
        ),
        iterations=args.iterations,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    args.markdown.write_text(
        render_markdown(audit),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": audit["status"],
        "decision": audit["decision"],
        "output": str(args.output),
        "markdown": str(args.markdown),
    }, indent=2))


if __name__ == "__main__":
    main()
