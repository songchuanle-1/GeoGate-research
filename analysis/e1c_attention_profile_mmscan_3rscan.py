#!/usr/bin/env python3
"""Profile last-prompt spatial attention on the frozen MMScan 3RScan subset.

This is a single-forward, read-only mechanism check. It does not load a
pruning hook, generate answers, or inspect reference answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IMAGE_TOKEN_INDEX,
    LOC_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_videos,
    tokenizer_special_token,
)
from llava.model.builder import load_pretrained_model

from adaptive_token.gpu_guard import guard_gpu, gpu_id_from_environment
from analysis.experiment_lock import lock_reference, require_lock


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = (
    ROOT
    / "results/raw/followup/mmscan_3rscan_confirmation_20260917"
    / "experiment_lock_v3.json"
)


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_digest(values: list[str]) -> str:
    return hashlib.sha256(
        "".join(f"{value}\n" for value in values).encode("utf-8")
    ).hexdigest()


def select_scene_samples(rows: list[dict]) -> list[dict]:
    selected = []
    seen = set()
    for row in rows:
        video = str(row["video"])
        if video in seen:
            continue
        seen.add(video)
        selected.append(row)
    return selected


def build_prompt(row: dict, conv_mode: str):
    prompt = str(row["conversations"][0]["value"])
    boxes = row.get("target", {}).get("boxes", [])
    if boxes:
        clicks = torch.tensor([
            [round(float(value), 3) for value in box[:3]]
            for box in boxes
        ])
        placeholder = ", ".join("<boxes>" for _ in boxes)
        prompt = prompt.replace("<boxes>", placeholder)
    else:
        clicks = torch.zeros((0, 3))

    prompt = prompt.replace(DEFAULT_VIDEO_TOKEN, DEFAULT_IMAGE_TOKEN)
    conversation = conv_templates[conv_mode].copy()
    conversation.append_message(conversation.roles[0], prompt)
    conversation.append_message(conversation.roles[1], None)
    return conversation.get_prompt(), clicks


def visual_span(input_ids: torch.Tensor, total_length: int) -> tuple[int, int]:
    """Return the prepared-embedding span occupied by the image features."""
    row = input_ids[0]
    image_positions = (row == IMAGE_TOKEN_INDEX).nonzero()
    if len(image_positions) != 1:
        raise ValueError("Expected exactly one image token")
    image_position = int(image_positions[0, 0])
    image_count = int((row == IMAGE_TOKEN_INDEX).sum().item())
    start = int(
        (row[:image_position] >= 0).sum().item()
        + (row[:image_position] == LOC_TOKEN_INDEX).sum().item()
    )
    # LOC tokens are one embedding each in both lengths, so they cancel.
    n_visual = total_length - len(row) + image_count
    if n_visual <= 0:
        raise ValueError("Prepared image span is empty")
    return start, start + n_visual


def half_mass(values: torch.Tensor) -> tuple[float, int]:
    mass = values.sum()
    if float(mass) <= 0:
        return float("nan"), 0
    ordered = torch.sort(values, descending=True).values
    cumulative = torch.cumsum(ordered, dim=0) / mass
    count = int((cumulative < 0.5).sum().item()) + 1
    return count / values.numel(), count


def bootstrap_mean(
    values: list[float],
    scenes: list[str],
    *,
    iterations: int,
    seed: int,
) -> tuple[float, tuple[float, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for scene, value in zip(scenes, values):
        grouped[scene].append(value)
    scene_ids = sorted(grouped)
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        selected = [scene_ids[rng.randrange(len(scene_ids))]
                    for _ in scene_ids]
        samples.append(
            sum(value for scene in selected for value in grouped[scene])
            / sum(len(grouped[scene]) for scene in selected)
        )

    def percentile(quantile: float) -> float:
        ordered = sorted(samples)
        index = round(quantile * (len(ordered) - 1))
        return ordered[min(len(ordered) - 1, max(0, index))]

    return sum(values) / len(values), (
        percentile(0.025),
        percentile(0.975),
    )


def summarize(
    per_sample: list[dict],
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict:
    scenes = [str(item["video"]) for item in per_sample]
    late = list(range(14, 27))
    middle = list(range(2, 14))
    endpoints = {}

    def endpoint(name: str, values: list[float]) -> None:
        mean, interval = bootstrap_mean(
            values,
            scenes,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        )
        endpoints[name] = {
            "mean": mean,
            "scene_95ci": list(interval),
        }

    endpoint(
        "late_minus_middle_attention_fraction",
        [
            float(np.mean([item["frac"][layer] for layer in late]))
            - float(np.mean([item["frac"][layer] for layer in middle]))
            for item in per_sample
        ],
    )
    endpoint(
        "l14_minus_l13_attention_fraction",
        [
            float(item["frac"][14]) - float(item["frac"][13])
            for item in per_sample
        ],
    )
    endpoint(
        "middle_minus_late_half_mass_coverage",
        [
            float(np.mean([item["top50"][layer] for layer in middle]))
            - float(np.mean([item["top50"][layer] for layer in late]))
            for item in per_sample
        ],
    )
    endpoint(
        "late_mean_half_mass_tokens",
        [
            float(np.mean([item["half_mass_tokens"][layer] for layer in late]))
            for item in per_sample
        ],
    )

    primary = endpoints["late_minus_middle_attention_fraction"]
    secondary = endpoints["l14_minus_l13_attention_fraction"]
    coverage = endpoints["middle_minus_late_half_mass_coverage"]
    passed = (
        primary["mean"] > 0
        and primary["scene_95ci"][0] > 0
        and secondary["mean"] > 0
        and secondary["scene_95ci"][0] > 0
        and coverage["mean"] > 0
        and coverage["scene_95ci"][0] > 0
    )
    return {
        "endpoints": endpoints,
        "primary_gate": {
            "passes": passed,
            "rule": (
                "The late-minus-middle attention fraction, L14-minus-L13 "
                "jump, and middle-minus-late half-mass coverage all have "
                "positive means and 95% scene-cluster lower bounds."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--question-file", type=Path, required=True)
    parser.add_argument("--video-folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--num-samples", type=int, default=234)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260919)
    args = parser.parse_args()

    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("Refusing to overwrite profile outputs")
    gpu_id = (
        int(args.gpu_id)
        if args.gpu_id is not None
        else gpu_id_from_environment()
    )
    allowed_pids = {os.getpid()}
    guard_gpu(gpu_id, allowed_pids, "preflight")
    lock = lock_reference(require_lock(args.lock, ROOT, "mmscan_3rscan"))
    metadata = lock["metadata"]
    if (
        int(metadata.get("questions", -1)) != 4_934
        or int(metadata.get("scenes", -1)) != 234
        or int(metadata.get("seed", -1)) != 0
        or int(metadata.get("num_frames", -1)) != 20
    ):
        raise RuntimeError("MMScan experiment lock metadata changed")

    rows = json.loads(args.question_file.read_text(encoding="utf-8"))
    selected = select_scene_samples(rows)[:args.num_samples]
    if len(selected) != args.num_samples:
        raise ValueError("Insufficient scene-stratified MMScan samples")
    question_ids = [str(row["prompt_id"]) for row in selected]
    videos = [str(row["video"]) for row in selected]
    if len(set(videos)) != len(videos):
        raise ValueError("Scene selection is not unique")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "geogate-mmscan-3rscan-attention-profile-protocol-v1",
        "status": "frozen",
        "model": "LLaVA-3D-7B",
        "dataset": "MMScan-QA-3RScan",
        "num_samples": len(selected),
        "scenes": len(set(videos)),
        "selection": "first question per video in official subset order",
        "question_ids_sha256": ordered_digest(question_ids),
        "video_ids_sha256": ordered_digest(videos),
        "num_frames": 20,
        "seed": args.seed,
        "conv_mode": args.conv_mode,
        "profile_code_sha256": digest_file(Path(__file__)),
        "question_file_sha256": digest_file(args.question_file),
        "experiment_lock": lock,
        "primary_rule": (
            "Late-minus-middle spatial-attention fraction, L14-minus-L13 "
            "jump, and middle-minus-late half-mass coverage must all have "
            "positive means and positive 95% scene-cluster lower bounds."
        ),
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    tokenizer, model, processor, _ = load_pretrained_model(
        args.model_path,
        None,
        get_model_name_from_path(args.model_path),
        device_map="cuda:0",
    )
    model.eval()
    if int(processor["video"].num_frames) != 20:
        raise RuntimeError("MMScan video processor is not in 20-frame mode")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    per_sample = []
    for index, row in enumerate(selected):
        guard_gpu(gpu_id, allowed_pids, f"question {index}")
        started = time.time()
        prompt, clicks = build_prompt(row, args.conv_mode)
        input_ids = tokenizer_special_token(
            prompt,
            tokenizer,
            return_tensors="pt",
        ).unsqueeze(0).cuda()
        videos_tensor = process_videos(
            str(args.video_folder / str(row["video"])),
            processor["video"],
            mode="random",
            device=model.device,
            text=row["conversations"][0]["value"],
        )
        tensors = {
            key: videos_tensor[key].to(
                model.device,
                dtype=torch.bfloat16,
            )
            for key in ("images", "depths", "poses", "intrinsics")
        }
        clicks = clicks.to(model.device, dtype=torch.bfloat16)
        with torch.inference_mode():
            prepared = model.prepare_inputs_labels_for_multimodal(
                input_ids,
                None,
                None,
                None,
                None,
                tensors["images"],
                tensors["depths"],
                tensors["poses"],
                tensors["intrinsics"],
                None,
                clicks=clicks,
                image_sizes=None,
            )
            position_ids = prepared[1]
            attention_mask = prepared[2]
            inputs_embeds = prepared[4]
            if inputs_embeds is None:
                raise RuntimeError("Multimodal preparation returned no embeddings")
            start, end = visual_span(input_ids, inputs_embeds.shape[1])
            output = model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )

        frac = []
        top50 = []
        top90 = []
        half_tokens = []
        gini = []
        visual_count = end - start
        for attention in output.attentions:
            matrix = attention[0].float().mean(dim=0)
            generation_row = matrix[-1]
            spatial = generation_row[start:end]
            fraction = float(spatial.sum() / generation_row.sum())
            coverage50, half_count = half_mass(spatial)
            ordered = torch.sort(spatial, descending=True).values
            cumulative = torch.cumsum(ordered, dim=0) / ordered.sum()
            coverage90 = (
                int((cumulative < 0.9).sum().item()) + 1
            ) / visual_count
            received = torch.sort(matrix[:, start:end].sum(dim=0)).values
            positions = torch.arange(
                1,
                len(received) + 1,
                device=received.device,
                dtype=received.dtype,
            )
            gini_value = (
                ((2 * positions - len(received) - 1) * received).sum()
                / (len(received) * received.sum() + 1e-9)
            ).item()
            frac.append(round(fraction, 6))
            top50.append(round(coverage50, 6))
            top90.append(round(coverage90, 6))
            half_tokens.append(half_count)
            gini.append(round(gini_value, 6))
        per_sample.append({
            "question_id": str(row["prompt_id"]),
            "video": str(row["video"]),
            "question": str(row["conversations"][0]["value"]),
            "visual_tokens": visual_count,
            "visual_span": [start, end],
            "latency_s": round(time.time() - started, 6),
            "frac": frac,
            "top50": top50,
            "top90": top90,
            "half_mass_tokens": half_tokens,
            "gini": gini,
        })
        del output
        torch.cuda.empty_cache()
        print(
            f"[{index + 1}/{len(selected)}] {row['video']} "
            f"visual={visual_count}",
            flush=True,
        )

    summary = summarize(
        per_sample,
        bootstrap_iterations=args.iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    output = {
        "schema": "geogate-mmscan-3rscan-attention-profile-v1",
        "state": "complete",
        "manifest": manifest,
        "summary": summary,
        "per_sample": per_sample,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
