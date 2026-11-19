#!/usr/bin/env python3
"""Probe mixed-scene heterogeneous GeoGate batching against per-sample runs."""

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

import numpy as np
import torch

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_videos,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model

from adaptive_token.geogate_batch_hetero_v18 import (
    enable_auto_spans,
    enable_geogate_batch_hetero,
)
from analysis.hetero_batch_eos_v18 import (
    raw_mismatch_is_eos_padding_only,
    trim_after_eos,
)
from analysis.hetero_batch_schedule_v18 import (
    build_workload_schedule,
    schedule_summary,
)


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def tensor_sha256(tensor):
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy()
    return hashlib.sha256(raw.tobytes()).hexdigest()


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


def build_prompt(question):
    conversation = conv_templates["llava_v1"].copy()
    conversation.append_message(
        conversation.roles[0],
        DEFAULT_IMAGE_TOKEN + "\n" + question,
    )
    conversation.append_message(conversation.roles[1], None)
    return conversation.get_prompt()


def left_pad(rows, pad_token_id, device):
    width = max(row.numel() for row in rows)
    input_ids = torch.full(
        (len(rows), width),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, row in enumerate(rows):
        input_ids[index, -row.numel():] = row.to(device)
        attention_mask[index, -row.numel():] = 1
    return input_ids, attention_mask


def tensor_dict_sha256(tensors):
    return {
        key: tensor_sha256(value)
        for key, value in tensors.items()
    }


def generate_batch(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    tensors,
    max_new_tokens,
):
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            images=tensors["images"],
            depths=tensors["depths"],
            poses=tensors["poses"],
            intrinsics=tensors["intrinsics"],
            image_sizes=None,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start
    outputs = output_ids.detach().cpu()
    output_rows = outputs.tolist()
    eos_token_id = tokenizer.eos_token_id
    return {
        "elapsed_s": elapsed_s,
        "output_ids": output_rows,
        "eos_trimmed_output_ids": [
            trim_after_eos(row, eos_token_id)
            for row in output_rows
        ],
        "eos_token_id": eos_token_id,
        "texts": tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        ),
        "allocated_before_mib": allocated_before / (1024 ** 2),
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--question-file", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--video-folder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument(
        "--workload",
        choices=("mixed", "length", "worst"),
        default="mixed",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    for path in (args.out, args.manifest, args.status):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    if args.batch_size < 2 or args.max_new_tokens < 1:
        raise ValueError("Probe requires batch size >=2 and positive tokens")
    guard_gpu(args.gpu_id, set(), "preflight")

    questions = json.loads(args.question_file.read_text(encoding="utf-8"))
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    schedule = build_workload_schedule(
        questions,
        inventory,
        args.workload,
        args.batch_size,
        args.seed,
    )
    if not 0 <= args.batch_index < len(schedule):
        raise ValueError("Batch index is outside the schedule")
    batch = schedule[args.batch_index]
    if not batch["cross_scene"]:
        raise ValueError("Selected batch is not cross-scene")
    if batch["visual_gap"] <= 0:
        raise ValueError("Selected batch does not contain variable visual counts")

    seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    tokenizer, baseline, processor, _ = load_pretrained_model(
        args.model_path,
        None,
        get_model_name_from_path(args.model_path),
        device_map="cuda:0",
    )
    baseline.eval()
    baseline.config.tokenizer_padding_side = "left"
    tokenizer.padding_side = "left"

    geogate_tokenizer, geogate, _, _ = load_pretrained_model(
        args.model_path,
        None,
        get_model_name_from_path(args.model_path),
        device_map="cuda:0",
    )
    geogate.eval()
    geogate.config.tokenizer_padding_side = "left"
    geogate_tokenizer.padding_side = "left"
    enable_geogate_batch_hetero(
        geogate,
        prune_layer=14,
        keep_ratio=0.10,
    )
    enable_auto_spans(geogate, IMAGE_TOKEN_INDEX)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer has neither pad nor EOS token")

    question_by_id = {
        str(row["question_id"]): row
        for row in questions
    }
    token_rows = []
    per_sample_tensors = []
    slot_metadata = []
    for slot in batch["slots"]:
        question_row = question_by_id[slot["question_id"]]
        prompt = build_prompt(question_row["text"])
        token_rows.append(tokenizer_image_token(
            prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ))
        seed_all(int(slot["scene_seed"]))
        video = process_videos(
            str(args.video_folder / slot["scene_id"]),
            processor["video"],
            mode="random",
            device=baseline.device,
            text=question_row["text"],
        )
        sample_tensors = {
            key: video[key].to(
                baseline.device,
                dtype=torch.bfloat16,
            )
            for key in ("images", "depths", "poses", "intrinsics")
        }
        per_sample_tensors.append(sample_tensors)
        slot_metadata.append({
            **slot,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "tensor_sha256": tensor_dict_sha256(sample_tensors),
        })

    input_ids, attention_mask = left_pad(
        token_rows,
        pad_token_id,
        baseline.device,
    )
    batch_tensors = {
        key: torch.cat(
            [sample[key] for sample in per_sample_tensors],
            dim=0,
        )
        for key in ("images", "depths", "poses", "intrinsics")
    }
    common = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "mixed-scene heterogeneous GeoGate batching correctness probe; "
            "timing is descriptive only"
        ),
        "model": {
            "path": str(Path(args.model_path).resolve()),
            "config_sha256": digest_file(
                Path(args.model_path) / "config.json"
            ),
        },
        "question_file": {
            "path": str(args.question_file.resolve()),
            "sha256": digest_file(args.question_file),
        },
        "inventory": {
            "path": str(args.inventory.resolve()),
            "sha256": digest_file(args.inventory),
        },
        "protocol": {
            "protocol_id": "hetero_batch_v18b_eos_trimmed",
            "workload": args.workload,
            "batch_size": args.batch_size,
            "batch_index": args.batch_index,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "greedy": True,
            "padding_side": "left",
            "prune_layer": 14,
            "keep_ratio": 0.10,
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
        },
        "schedule_summary": schedule_summary(schedule),
        "batch": {
            "scene_count": batch["scene_count"],
            "cross_scene": batch["cross_scene"],
            "visual_min": batch["visual_min"],
            "visual_max": batch["visual_max"],
            "visual_gap": batch["visual_gap"],
            "slots": slot_metadata,
        },
        "inputs": {
            "input_ids_shape": list(input_ids.shape),
            "attention_mask_shape": list(attention_mask.shape),
            "input_lengths": [int(row.numel()) for row in token_rows],
            "tensor_shapes": {
                key: list(value.shape)
                for key, value in batch_tensors.items()
            },
            "tensor_sha256": tensor_dict_sha256(batch_tensors),
        },
        "provenance": {
            "runner_sha256": digest_file(__file__),
            "batch_patch_sha256": digest_file(
                inspect.getfile(enable_geogate_batch_hetero)
            ),
            "schedule_sha256": digest_file(
                Path(__file__).with_name("hetero_batch_schedule_v18.py")
            ),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("x", encoding="utf-8") as handle:
        json.dump(common, handle, indent=2, sort_keys=True)
        handle.write("\n")

    allowed_pids = {os.getpid()}
    guard_gpu(args.gpu_id, allowed_pids, "baseline batch")
    baseline_batch = generate_batch(
        baseline,
        tokenizer,
        input_ids,
        attention_mask,
        batch_tensors,
        args.max_new_tokens,
    )

    guard_gpu(args.gpu_id, allowed_pids, "GeoGate batch")
    geogate_batch = generate_batch(
        geogate,
        geogate_tokenizer,
        input_ids,
        attention_mask,
        batch_tensors,
        args.max_new_tokens,
    )
    batch_trace = list(geogate.model._gg_hetero_trace)

    individual = []
    for sample_index, slot in enumerate(batch["slots"]):
        guard_gpu(
            args.gpu_id,
            allowed_pids,
            f"GeoGate individual {sample_index}",
        )
        single_ids = input_ids[sample_index:sample_index + 1]
        single_mask = attention_mask[sample_index:sample_index + 1]
        single_tensors = {
            key: value[sample_index:sample_index + 1]
            for key, value in batch_tensors.items()
        }
        result = generate_batch(
            geogate,
            geogate_tokenizer,
            single_ids,
            single_mask,
            single_tensors,
            args.max_new_tokens,
        )
        individual.append({
            "sample_index": sample_index,
            "question_id": slot["question_id"],
            **result,
            "trace": list(geogate.model._gg_hetero_trace),
        })

    raw_exact = [
        geogate_batch["output_ids"][index] == row["output_ids"][0]
        for index, row in enumerate(individual)
    ]
    eos_trimmed_exact = [
        geogate_batch["eos_trimmed_output_ids"][index]
        == row["eos_trimmed_output_ids"][0]
        for index, row in enumerate(individual)
    ]
    padding_only = [
        raw_mismatch_is_eos_padding_only(
            geogate_batch["output_ids"][index],
            row["output_ids"][0],
            geogate_batch["eos_token_id"],
            pad_token_id,
        )
        for index, row in enumerate(individual)
    ]
    event = batch_trace[0] if len(batch_trace) == 1 else {}
    expected_after = [
        int(event.get("before", -1))
        - (int(visual) - int(keep))
        for visual, keep in zip(
            event.get("visual_counts", []),
            event.get("keep_counts", []),
        )
    ]
    trace_valid = (
        len(batch_trace) == 1
        and event.get("event") == "prune"
        and event.get("layer") == 14
        and event.get("batch") == args.batch_size
        and event.get("visual_counts")
        == [slot["visual_tokens"] for slot in batch["slots"]]
        and len(set(event.get("visual_counts", []))) > 1
        and event.get("valid_after") == expected_after
    )
    passed = all(eos_trimmed_exact) and trace_valid
    result = {
        "state": "passed" if passed else "failed",
        "batch_trace": batch_trace,
        "baseline_batch": baseline_batch,
        "geogate_batch": geogate_batch,
        "geogate_individual": individual,
        "correctness": {
            "raw_exact_token_match_per_sample": raw_exact,
            "all_raw_exact": all(raw_exact),
            "eos_trimmed_exact_token_match_per_sample": eos_trimmed_exact,
            "all_eos_trimmed_exact": all(eos_trimmed_exact),
            "raw_mismatch_is_eos_padding_only_per_sample": padding_only,
            "all_raw_mismatches_are_eos_padding_only": all(
                exact or padding
                for exact, padding in zip(raw_exact, padding_only)
            ),
            "trace_valid": trace_valid,
        },
    }
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    status = {
        "state": "finished",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "raw_sha256": digest_file(args.out),
        "manifest_sha256": digest_file(args.manifest),
    }
    with args.status.open("x", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(status, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
