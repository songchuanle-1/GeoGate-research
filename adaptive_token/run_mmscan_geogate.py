#!/usr/bin/env python3
"""Run the frozen MMScan-QA paired evaluation with GeoGate.

This mirrors the official LLaVA-3D ``model_mmscan_qa.py`` input path,
including target boxes and ``tokenizer_special_token``, while adding the
same fixed layer/ratio pruning implementation used in the ScanQA runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_videos,
    tokenizer_special_token,
)
from llava.model.builder import load_pretrained_model

from adaptive_token.geogate_v0 import (
    enable_auto_span,
    enable_geogate,
)
from adaptive_token.gpu_guard import (
    gpu_id_from_environment,
    guard_gpu,
)
from analysis.experiment_lock import lock_reference, require_lock


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = (
    ROOT
    / "results/raw/followup/mmscan_3rscan_confirmation_20260917"
    / "experiment_lock_v3.json"
)


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor metadata and bytes without relying on pickle."""
    value = tensor.detach().to("cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


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


def chunk_rows(rows: list[dict], chunks: int, index: int) -> list[dict]:
    if chunks < 1 or not 0 <= index < chunks:
        raise ValueError("Invalid chunk configuration")
    size = (len(rows) + chunks - 1) // chunks
    return rows[index * size:(index + 1) * size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--video-folder", required=True)
    parser.add_argument("--answers-file", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--prune-layer", type=int, default=-1)
    parser.add_argument("--keep-ratio", type=float, default=0.10)
    parser.add_argument("--scorer", default="attn")
    parser.add_argument("--prune-mode", choices=("keep_topk", "delete_topk"),
                        default="keep_topk")
    parser.add_argument("--delete-n", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()

    lock = lock_reference(
        require_lock(args.lock, ROOT, "mmscan_3rscan")
    )
    metadata = lock["metadata"]
    if (
        int(metadata.get("questions", -1)) != 4_934
        or int(metadata.get("scenes", -1)) != 234
        or int(metadata.get("seed", -1)) != 0
        or int(metadata.get("num_frames", -1)) != 20
        or int(metadata.get("condition", {}).get("prune_layer", -1)) != 14
        or abs(
            float(metadata.get("condition", {}).get("keep_ratio", -1))
            - 0.10
        ) > 1e-9
    ):
        raise RuntimeError("MMScan experiment lock metadata changed")
    gpu_id = (
        int(args.gpu_id)
        if args.gpu_id is not None
        else gpu_id_from_environment()
    )
    allowed_pids = {os.getpid()}
    guard_gpu(gpu_id, allowed_pids, "preflight")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows = json.loads(open(args.question_file, encoding="utf-8").read())
    rows = chunk_rows(rows, args.num_chunks, args.chunk_idx)
    tokenizer, model, processor, _ = load_pretrained_model(
        args.model_path,
        None,
        get_model_name_from_path(args.model_path),
        device_map="cuda:0",
    )
    model.eval()
    if int(processor["video"].num_frames) != int(metadata["num_frames"]):
        raise RuntimeError(
            "MMScan video processor frame count differs from the frozen lock"
        )

    enabled = args.prune_layer >= 0
    if enabled:
        enable_geogate(
            model,
            prune_layer=args.prune_layer,
            keep_ratio=args.keep_ratio,
            scorer=args.scorer,
            prune_mode=args.prune_mode,
            delete_n=args.delete_n,
        )
        enable_auto_span(model, IMAGE_TOKEN_INDEX)

    with open(args.answers_file, "w", encoding="utf-8") as output:
        for index, row in enumerate(tqdm(rows)):
            guard_gpu(gpu_id, allowed_pids, f"question {index}")
            prompt, clicks = build_prompt(row, args.conv_mode)
            input_ids = tokenizer_special_token(
                prompt,
                tokenizer,
                return_tensors="pt",
            ).unsqueeze(0).cuda()
            video_path = f"{args.video_folder}/{row['video']}"
            videos = process_videos(
                video_path,
                processor["video"],
                mode="random",
                device=model.device,
                text=row["conversations"][0]["value"],
            )
            tensors = {
                key: videos[key].to(model.device, dtype=torch.bfloat16)
                for key in ("images", "depths", "poses", "intrinsics")
            }
            input_tensor_hashes = {
                key: tensor_sha256(value)
                for key, value in tensors.items()
            }
            input_ids_sha256 = tensor_sha256(input_ids)
            clicks = clicks.to(model.device, dtype=torch.bfloat16)
            clicks_sha256 = tensor_sha256(clicks)
            torch.cuda.synchronize()
            started = time.time()
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=tensors["images"],
                    depths=tensors["depths"],
                    poses=tensors["poses"],
                    intrinsics=tensors["intrinsics"],
                    clicks=clicks,
                    image_sizes=None,
                    do_sample=False,
                    temperature=0,
                    num_beams=1,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
            torch.cuda.synchronize()
            elapsed = time.time() - started
            prediction = tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
            )[0].strip()
            output.write(json.dumps({
                "question_id": str(row["prompt_id"]),
                "prompt": str(row["conversations"][0]["value"]),
                "pred": [prediction],
                "gt": row["conversations"][1]["value"],
                "latency_s": round(elapsed, 6),
                "input_tensor_hashes": input_tensor_hashes,
                "input_tensor_shapes": {
                    key: list(value.shape)
                    for key, value in tensors.items()
                },
                "input_ids_sha256": input_ids_sha256,
                "clicks_sha256": clicks_sha256,
                "prune_layer": int(args.prune_layer),
                "keep_ratio": float(args.keep_ratio),
                "pruner": "geogate" if enabled else "baseline",
                "scorer": args.scorer if enabled else None,
                "seed": int(args.seed),
                "experiment_lock": lock,
            }) + "\n")
            output.flush()


if __name__ == "__main__":
    main()
