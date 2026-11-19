#!/usr/bin/env python3
"""Run one frozen full ScanQA scene-bucketed throughput session."""

import argparse
from datetime import datetime, timezone
import hashlib
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

from adaptive_token.geogate_batch_v6 import (
    enable_auto_spans,
    enable_geogate_batch,
)
from analysis.batch_throughput_v7_common import (
    batch_size_order,
    build_scene_batches,
    condition_order,
    schedule_summary,
    validate_inventory,
)


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
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


def prepare_batch(batch, tokenizer, processor, model, video_folder):
    rows = batch["slots"]
    token_rows = [
        tokenizer_image_token(
            build_prompt(row["text"]),
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )
        for row in rows
    ]
    seed_all(batch["scene_seed"])
    video = process_videos(
        str(Path(video_folder) / batch["scene_id"]),
        processor["video"],
        mode="random",
        device=model.device,
        text=rows[0]["text"],
    )
    tensors = {
        key: torch.cat(
            [video[key].to(model.device, dtype=torch.bfloat16)] * len(rows),
            dim=0,
        )
        for key in ("images", "depths", "poses", "intrinsics")
    }
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer has neither pad nor EOS token")
    input_ids, attention_mask = left_pad(
        token_rows,
        pad_token_id,
        model.device,
    )
    return input_ids, attention_mask, tensors


def generate(
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
    outputs = output_ids.detach().cpu().tolist()
    return {
        "elapsed_s": elapsed_s,
        "output_ids": outputs,
        "texts": tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
        ),
        "allocated_before_mib": allocated_before / (1024 ** 2),
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
        ),
        "executed_generated_tokens": sum(map(len, outputs)),
    }


def write_rows(handle, rows):
    for row in rows:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    handle.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--video-folder", required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--session-index", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--only-scene", action="append")
    args = parser.parse_args()

    for path in (args.out, args.manifest, args.status):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    if args.max_new_tokens != 512:
        raise ValueError("Frozen v7 protocol requires max_new_tokens=512")
    guard_gpu(args.gpu_id, set(), "preflight")

    seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    questions = json.loads(Path(args.question_file).read_text(encoding="utf-8"))
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    grouped, _ = validate_inventory(questions, inventory)
    selected_scenes = args.only_scene or list(grouped)
    schedules = {
        batch_size: build_scene_batches(
            questions,
            inventory,
            batch_size,
            selected_scenes,
        )
        for batch_size in (2, 4)
    }
    summaries = {
        str(batch_size): schedule_summary(schedules[batch_size])
        for batch_size in (2, 4)
    }
    selected_question_ids = [
        str(row["question_id"])
        for scene_id, rows in grouped.items()
        if scene_id in set(selected_scenes)
        for row in rows
    ]
    is_full = len(selected_question_ids) == 4675 and len(selected_scenes) == 71
    if is_full:
        expected = {
            "2": {
                "real_questions": 4675,
                "padding_slots": 35,
                "executed_slots": 4710,
            },
            "4": {
                "real_questions": 4675,
                "padding_slots": 101,
                "executed_slots": 4776,
            },
        }
        for key, values in expected.items():
            for field, value in values.items():
                if summaries[key][field] != value:
                    raise ValueError(
                        f"Frozen full schedule mismatch: B{key} "
                        f"{field}={summaries[key][field]} expected {value}"
                    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "confirmatory full ScanQA scene-bucketed tensor-batch generation "
            "throughput" if is_full else
            "construction and resource smoke for the frozen v7 runner"
        ),
        "session_index": args.session_index,
        "is_full": is_full,
        "model": {
            "path": str(Path(args.model_path).resolve()),
            "config_sha256": digest_file(Path(args.model_path) / "config.json"),
        },
        "questions": {
            "path": str(Path(args.question_file).resolve()),
            "sha256": digest_file(args.question_file),
            "count": len(selected_question_ids),
            "scene_count": len(selected_scenes),
            "ids": selected_question_ids,
            "scenes": selected_scenes,
        },
        "inventory": {
            "path": str(args.inventory.resolve()),
            "sha256": digest_file(args.inventory),
            "runner_sha256": inventory["runner_sha256"],
        },
        "protocol": {
            "batch_sizes": [2, 4],
            "batch_size_order": list(batch_size_order(args.session_index)),
            "bucketing": "scene only, official scene and question order",
            "remainder": (
                "duplicate the final real question within each scene; include "
                "its execution in elapsed time but exclude it from useful "
                "question/token counts and quality"
            ),
            "schedule_summary": summaries,
            "warmup_batches_per_size": args.warmup_batches,
            "warmup_rows_emitted": False,
            "max_new_tokens": args.max_new_tokens,
            "padding_side": "left",
            "greedy": True,
            "seed": args.seed,
            "scene_seed": "seed + official-order scene index from inventory",
            "condition_order": (
                "(batch_index + session_index + I[B=4]) parity"
            ),
            "cuda_synchronize": True,
            "timing_boundary": "generate with resident GPU RGB-D tensors",
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
            "geogate": {
                "prune_layer": 14,
                "keep_ratio": 0.10,
                "scorer": "last_prompt_attention",
            },
            "bootstrap_unit": "scene",
            "sessions_pooled": False,
            "promotion_gate": (
                "all three full sessions protocol-clean; for a promoted batch "
                "size, each session has ratio >=1.03 and scene-bootstrap 95% "
                "lower bound >1; fixed-protocol EOS-trimmed tokens reproduce "
                "exactly across sessions; quality is reported without an "
                "equivalence claim"
            ),
        },
        "provenance": {
            "runner_sha256": digest_file(__file__),
            "common_sha256": digest_file(
                Path(__file__).with_name("batch_throughput_v7_common.py")
            ),
            "batch_patch_sha256": digest_file(
                enable_geogate_batch.__code__.co_filename
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

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
    enable_geogate_batch(geogate, prune_layer=14, keep_ratio=0.10)
    enable_auto_spans(geogate, IMAGE_TOKEN_INDEX)

    allowed_pids = {os.getpid()}
    measured_rows = 0
    measured_batches = 0
    with args.out.open("x", encoding="utf-8") as output:
        for batch_size in batch_size_order(args.session_index):
            batches = schedules[batch_size]
            for warmup_index, batch in enumerate(
                batches[:args.warmup_batches]
            ):
                guard_gpu(
                    args.gpu_id,
                    allowed_pids,
                    f"B{batch_size} warmup {warmup_index}",
                )
                prepared = prepare_batch(
                    batch,
                    tokenizer,
                    processor,
                    baseline,
                    args.video_folder,
                )
                for condition in condition_order(
                    warmup_index,
                    batch_size,
                    args.session_index,
                ):
                    model = baseline if condition == "baseline" else geogate
                    condition_tokenizer = (
                        tokenizer
                        if condition == "baseline"
                        else geogate_tokenizer
                    )
                    generate(
                        model,
                        condition_tokenizer,
                        *prepared,
                        args.max_new_tokens,
                    )

            for batch in batches:
                batch_index = batch["batch_index"]
                guard_gpu(
                    args.gpu_id,
                    allowed_pids,
                    f"B{batch_size} measured {batch_index}",
                )
                input_ids, attention_mask, tensors = prepare_batch(
                    batch,
                    tokenizer,
                    processor,
                    baseline,
                    args.video_folder,
                )
                for order, condition in enumerate(condition_order(
                    batch_index,
                    batch_size,
                    args.session_index,
                )):
                    model = baseline if condition == "baseline" else geogate
                    condition_tokenizer = (
                        tokenizer
                        if condition == "baseline"
                        else geogate_tokenizer
                    )
                    result = generate(
                        model,
                        condition_tokenizer,
                        input_ids,
                        attention_mask,
                        tensors,
                        args.max_new_tokens,
                    )
                    useful_tokens = sum(
                        len(result["output_ids"][sample_index])
                        for sample_index, row in enumerate(batch["slots"])
                        if not row["_is_padding"]
                    )
                    output_rows = []
                    for sample_index, row in enumerate(batch["slots"]):
                        output_rows.append({
                            "session_index": args.session_index,
                            "batch_size": batch_size,
                            "batch_index": batch_index,
                            "scene_batch_index": batch["scene_batch_index"],
                            "sample_index": sample_index,
                            "condition": condition,
                            "order": order,
                            "question_id": str(row["question_id"]),
                            "source_question_id": row["_source_question_id"],
                            "question_index": row["_question_index"],
                            "scene_id": batch["scene_id"],
                            "scene_index": batch["scene_index"],
                            "scene_seed": batch["scene_seed"],
                            "visual_tokens": batch["visual_tokens"],
                            "is_padding": row["_is_padding"],
                            "real_questions_in_batch": batch["real_count"],
                            "padding_slots_in_batch": batch["padding_count"],
                            "elapsed_batch_s": result["elapsed_s"],
                            "useful_queries_per_s": (
                                batch["real_count"] / result["elapsed_s"]
                            ),
                            "executed_slots_per_s": (
                                batch_size / result["elapsed_s"]
                            ),
                            "generated_tokens": len(
                                result["output_ids"][sample_index]
                            ),
                            "useful_generated_tokens_in_batch": useful_tokens,
                            "executed_generated_tokens_in_batch": (
                                result["executed_generated_tokens"]
                            ),
                            "output_ids": result["output_ids"][sample_index],
                            "text": result["texts"][sample_index],
                            "allocated_before_mib": (
                                result["allocated_before_mib"]
                            ),
                            "peak_allocated_mib": (
                                result["peak_allocated_mib"]
                            ),
                            "incremental_peak_mib": (
                                result["peak_allocated_mib"]
                                - result["allocated_before_mib"]
                            ),
                            "geogate_trace": (
                                geogate.model._gg_batch_trace
                                if condition == "geogate"
                                else None
                            ),
                        })
                    write_rows(output, output_rows)
                    measured_rows += len(output_rows)
                    measured_batches += 1

    expected_rows = 2 * sum(
        summary["executed_slots"] for summary in summaries.values()
    )
    expected_measured_batches = 2 * sum(
        summary["batches"] for summary in summaries.values()
    )
    if measured_rows != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows, wrote {measured_rows}"
        )
    if measured_batches != expected_measured_batches:
        raise RuntimeError(
            f"Expected {expected_measured_batches} condition-batches, "
            f"measured {measured_batches}"
        )
    status = {
        "state": "finished",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "session_index": args.session_index,
        "is_full": is_full,
        "questions": len(selected_question_ids),
        "scenes": len(selected_scenes),
        "conditions": 2,
        "batch_sizes": [2, 4],
        "rows": measured_rows,
        "condition_batches": measured_batches,
        "raw_sha256": digest_file(args.out),
        "manifest_sha256": digest_file(args.manifest),
    }
    with args.status.open("x", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
