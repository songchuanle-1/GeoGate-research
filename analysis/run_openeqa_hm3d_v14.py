#!/usr/bin/env python3
"""Run the frozen paired OpenEQA-HM3D v14 protocol.

This runner is intentionally not launched until the authorized HM3D RGB-D,
pose, and intrinsic assets pass the preflight checks in the frozen protocol.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import time
import types

import numpy as np
import torch

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model

from adaptive_token.deterministic_scene_cap import (
    enable_deterministic_scene_cap,
    set_deterministic_scene_cap,
)
from adaptive_token.geogate_v0 import enable_geogate
from adaptive_token.openeqa_hm3d_v14 import (
    digest_file,
    load_hm3d_episode,
    resolve_episode_dir,
    tensor_sha256,
)


FROZEN_SEED = 20260913
FROZEN_QUESTIONS = 557
FROZEN_EPISODES = 63
FROZEN_FRAMES = 20
FROZEN_LAYER = 14
FROZEN_KEEP_RATIO = 0.10


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_pids(gpu_id):
    output = subprocess.check_output([
        "nvidia-smi",
        "-i",
        str(gpu_id),
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ], text=True)
    return {
        int(line.strip())
        for line in output.splitlines()
        if line.strip()
    }


def guard_gpu(gpu_id, allowed_pids, context):
    foreign = compute_pids(gpu_id) - set(allowed_pids)
    if foreign:
        raise RuntimeError(
            f"{context}: foreign GPU PIDs on device {gpu_id}: "
            f"{sorted(foreign)}"
        )


def build_prompt(question, conv_mode):
    conversation = conv_templates[conv_mode].copy()
    conversation.append_message(
        conversation.roles[0],
        DEFAULT_IMAGE_TOKEN + "\n" + question,
    )
    conversation.append_message(conversation.roles[1], None)
    return conversation.get_prompt()


def selected_questions(dataset, questions_per_episode):
    selected = [
        row for row in dataset
        if str(row.get("episode_history", "")).startswith("hm3d-v0/")
    ]
    if questions_per_episode is None:
        return selected
    counts = {}
    smoke = []
    for row in selected:
        episode = str(row["episode_history"])
        count = counts.get(episode, 0)
        if count < questions_per_episode:
            smoke.append(row)
            counts[episode] = count + 1
    return smoke


def configure_baseline(model, native_forward):
    model.model.forward = native_forward
    model.model._gg_span = None
    model.model._gg_next_pos = None
    model.model._gg_stages = {}
    model.model._gg_attn_history = []


def configure_geogate(
    model,
    geogate_forward,
    spatial_start,
    spatial_tokens,
):
    model.model.forward = geogate_forward
    model.model._gg_span = (
        int(spatial_start),
        int(spatial_start + spatial_tokens),
    )
    model.model._gg_next_pos = None
    model.model._gg_prune_layer = FROZEN_LAYER
    model.model._gg_keep_ratio = FROZEN_KEEP_RATIO
    model.model._gg_stages = {FROZEN_LAYER: FROZEN_KEEP_RATIO}
    model.model._gg_stages_user_set = False
    model.model._gg_scorer = "attn"
    model.model._gg_attn_history = []


class PreparedInputCache:
    """Reuse one exact multimodal preparation across paired conditions."""

    def __init__(self, model):
        self.model = model
        self.original = model.prepare_inputs_labels_for_multimodal
        self.prepared = None
        self.input_ids = None
        self.prefill_hits = 0

        def wrapped(_model, *args, **kwargs):
            ids = args[0] if args else kwargs.get("input_ids")
            images = args[5] if len(args) > 5 else kwargs.get("images")
            is_prefill = (
                torch.is_tensor(ids)
                and ids.ndim == 2
                and ids.shape[1] > 1
                and images is not None
            )
            if not is_prefill:
                return self.original(*args, **kwargs)
            if self.prepared is None or self.input_ids is None:
                raise RuntimeError(
                    "Prepared input cache is not initialized"
                )
            if (
                ids.shape != self.input_ids.shape
                or not torch.equal(ids, self.input_ids)
            ):
                raise RuntimeError("Cached prompt token IDs differ")
            self.prefill_hits += 1
            return self.prepared

        self.wrapper = types.MethodType(wrapped, model)
        model.prepare_inputs_labels_for_multimodal = self.wrapper

    def prepare(self, input_ids, tensors):
        if self.prepared is not None:
            raise RuntimeError(
                "Clear the previous prepared input first"
            )
        with torch.inference_mode():
            prepared = self.original(
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
                None,
                None,
            )
        if prepared[4] is None:
            raise RuntimeError(
                "Multimodal preparation returned no embedding"
            )
        self.prepared = prepared
        self.input_ids = input_ids.clone()
        self.prefill_hits = 0
        return prepared

    def clear(self):
        self.prepared = None
        self.input_ids = None
        self.prefill_hits = 0


def generate(model, tokenizer, input_ids, tensors):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensors["images"],
            depths=tensors["depths"],
            poses=tensors["poses"],
            intrinsics=tensors["intrinsics"],
            clicks=torch.zeros(
                (0, 3),
                device=model.device,
                dtype=torch.bfloat16,
            ),
            image_sizes=None,
            do_sample=False,
            num_beams=1,
            max_new_tokens=512,
            use_cache=True,
        )
    torch.cuda.synchronize()
    return {
        "answer": tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )[0].strip(),
        "output_ids_sha256": tensor_sha256(output_ids),
        "latency_s": time.perf_counter() - start,
    }


def write_json_exclusive(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--questions-per-episode", type=int)
    args = parser.parse_args()

    for path in (args.out, args.manifest, args.status):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    if not args.episode_root.is_dir():
        raise FileNotFoundError(
            "Authorized OpenEQA-HM3D RGB-D assets are unavailable: "
            f"{args.episode_root}"
        )
    guard_gpu(args.gpu_id, set(), "preflight")

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    selected = selected_questions(dataset, args.questions_per_episode)
    episodes = list(dict.fromkeys(
        str(row["episode_history"]) for row in selected
    ))
    full = args.questions_per_episode is None
    if full and (
        len(selected) != FROZEN_QUESTIONS
        or len(episodes) != FROZEN_EPISODES
    ):
        raise ValueError(
            "Full execution must contain the frozen 557 questions "
            "and 63 episodes"
        )
    if len({str(row["question_id"]) for row in selected}) != len(selected):
        raise ValueError("Question IDs are not unique")

    seed_all(FROZEN_SEED)
    torch.use_deterministic_algorithms(True)
    tokenizer, model, processor, _ = load_pretrained_model(
        str(args.model_path),
        None,
        get_model_name_from_path(str(args.model_path)),
        device_map="cuda:0",
    )
    model.eval()
    if int(processor["video"].num_frames) != FROZEN_FRAMES:
        raise ValueError(
            f"Frozen v14 requires {FROZEN_FRAMES} frames, found "
            f"{processor['video'].num_frames}"
        )
    native_forward = model.model.forward
    enable_geogate(
        model,
        prune_layer=FROZEN_LAYER,
        keep_ratio=FROZEN_KEEP_RATIO,
        scorer="attn",
    )
    geogate_forward = model.model.forward
    enable_deterministic_scene_cap(model)
    preparation_cache = PreparedInputCache(model)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "full frozen OpenEQA-HM3D v14 paired confirmation"
            if full else
            "OpenEQA-HM3D v14 construction smoke"
        ),
        "is_full": full,
        "questions": len(selected),
        "episodes": episodes,
        "question_ids": [str(row["question_id"]) for row in selected],
        "model": {
            "path": str(args.model_path.resolve()),
            "config_sha256": digest_file(
                args.model_path / "config.json"
            ),
        },
        "dataset": {
            "path": str(args.dataset.resolve()),
            "sha256": digest_file(args.dataset),
        },
        "episode_root": str(args.episode_root.resolve()),
        "protocol": {
            "seed": FROZEN_SEED,
            "num_frames": FROZEN_FRAMES,
            "conditions": ["native", "geogate_L14_R010"],
            "condition_order": "native_then_geogate",
            "prompt": "<image>\\n{official_question} with llava_v1",
            "decoding": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": 512,
            },
            "frame_selection": (
                "deterministic uniform official extracted HM3D RGB-D frames"
            ),
            "prepared_input": (
                "one exact prepared embedding object reused by native and "
                "GeoGate prefill"
            ),
            "primary_metric": "official OpenEQA LLM-Match",
            "bootstrap_unit": "episode",
        },
        "provenance": {
            "runner_sha256": digest_file(__file__),
            "hm3d_adapter_sha256": digest_file(
                inspect.getfile(load_hm3d_episode)
            ),
            "preflight_sha256": digest_file(
                Path(__file__).with_name(
                    "preflight_openeqa_hm3d_v14.py"
                )
            ),
            "audit_sha256": digest_file(
                Path(__file__).with_name(
                    "audit_openeqa_hm3d_v14.py"
                )
            ),
            "scorer_sha256": digest_file(
                Path(__file__).with_name(
                    "score_openeqa_hm3d_v14.py"
                )
            ),
            "geogate_sha256": digest_file(
                inspect.getfile(enable_geogate)
            ),
            "scene_cap_sha256": digest_file(
                inspect.getfile(enable_deterministic_scene_cap)
            ),
        },
    }
    write_json_exclusive(args.manifest, manifest)

    allowed_pids = {os.getpid()}
    current_episode = None
    episode_tensors = None
    episode_trace = None
    completed = 0
    with args.out.open("x", encoding="utf-8") as handle:
        for index, row in enumerate(selected):
            guard_gpu(
                args.gpu_id,
                allowed_pids,
                f"question {index}",
            )
            episode = str(row["episode_history"])
            if episode != current_episode:
                episode_dir = resolve_episode_dir(
                    args.episode_root,
                    episode,
                )
                cpu_tensors, episode_trace = load_hm3d_episode(
                    episode_dir,
                    processor["video"],
                )
                episode_tensors = {
                    key: value.to(
                        model.device,
                        dtype=torch.bfloat16,
                    )
                    for key, value in cpu_tensors.items()
                }
                current_episode = episode

            prompt = build_prompt(
                str(row["question"]),
                args.conv_mode,
            )
            input_ids = tokenizer_image_token(
                prompt,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(model.device)
            image_positions = (
                input_ids[0] == IMAGE_TOKEN_INDEX
            ).nonzero(as_tuple=False).flatten()
            if image_positions.numel() != 1:
                raise RuntimeError(
                    "Expected exactly one image placeholder"
                )
            spatial_start = int(image_positions[0].item())

            preparation_cache.clear()
            set_deterministic_scene_cap(
                model,
                episode,
                FROZEN_SEED,
            )
            prepared = preparation_cache.prepare(
                input_ids,
                episode_tensors,
            )
            full_tokens = int(prepared[4].shape[1])
            spatial_tokens = (
                full_tokens - int(input_ids.shape[1]) + 1
            )
            if spatial_tokens < 1:
                raise RuntimeError(
                    "Prepared input contains no spatial tokens"
                )
            expanded_input_sha256 = tensor_sha256(prepared[4])
            preparation_cap = list(
                model._deterministic_scene_cap_trace
            )
            common = {
                "question_id": str(row["question_id"]),
                "episode": episode,
                "scene_id": episode.rsplit("-hm3d-", 1)[-1],
                "category": str(row.get("category", "unknown")),
                "question": str(row["question"]),
                "prompt": prompt,
                "frame_selection": episode_trace,
                "input_tensor_sha256": {
                    key: tensor_sha256(value)
                    for key, value in episode_tensors.items()
                },
                "expanded_input_sha256": expanded_input_sha256,
                "full_tokens": full_tokens,
                "spatial_start": spatial_start,
                "input_visual_tokens": spatial_tokens,
                "seed": FROZEN_SEED,
            }

            configure_baseline(model, native_forward)
            baseline = generate(
                model,
                tokenizer,
                input_ids,
                episode_tensors,
            )
            if tensor_sha256(prepared[4]) != expanded_input_sha256:
                raise RuntimeError(
                    "Baseline mutated the prepared embedding"
                )

            configure_geogate(
                model,
                geogate_forward,
                spatial_start,
                spatial_tokens,
            )
            geogate = generate(
                model,
                tokenizer,
                input_ids,
                episode_tensors,
            )
            if tensor_sha256(prepared[4]) != expanded_input_sha256:
                raise RuntimeError(
                    "GeoGate mutated the prepared embedding"
                )
            if preparation_cache.prefill_hits != 2:
                raise RuntimeError(
                    "Each condition must reuse the prepared prefill exactly "
                    "once"
                )

            for condition, result, forward_path in (
                (
                    "native",
                    baseline,
                    "native_unpatched_llama_forward",
                ),
                (
                    "geogate_L14_R010",
                    geogate,
                    "geogate_patched_llama_forward",
                ),
            ):
                record = {
                    **common,
                    "condition": condition,
                    "answer": result["answer"],
                    "output_ids_sha256": (
                        result["output_ids_sha256"]
                    ),
                    "latency_s": result["latency_s"],
                    "forward_path": forward_path,
                    "deterministic_scene_cap": preparation_cap,
                    "preparation_calls": 1,
                    "prepared_prefill_cache_hits": (
                        preparation_cache.prefill_hits
                    ),
                }
                handle.write(
                    json.dumps(record, sort_keys=True) + "\n"
                )
            handle.flush()
            completed += 1
            preparation_cache.clear()

    status = {
        "state": "completed_unscored",
        "completed_questions": completed,
        "expected_questions": len(selected),
        "raw_sha256": digest_file(args.out),
        "manifest_sha256": digest_file(args.manifest),
        "note": (
            "Do not inspect or score until the structural audit passes."
        ),
    }
    write_json_exclusive(args.status, status)


if __name__ == "__main__":
    main()
