#!/usr/bin/env python3
"""Run one full mixed-scene heterogeneous GeoGate throughput session."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import torch

from llava.constants import IMAGE_TOKEN_INDEX
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
from analysis.hetero_batch_schedule_v18 import (
    WORKLOADS,
    build_workload_schedule,
    condition_order,
    schedule_summary,
)
from analysis.run_hetero_batch_v18_probe import (
    build_prompt,
    digest_file,
    generate_batch,
    guard_gpu,
    left_pad,
    seed_all,
    tensor_dict_sha256,
)


FROZEN_SCENES = 71
FROZEN_QUESTIONS = 4675


def build_token_rows(batch, questions, tokenizer):
    rows = {
        str(row["question_id"]): row
        for row in questions
    }
    token_rows = []
    for slot in batch["slots"]:
        question = rows[slot["question_id"]]
        prompt = build_prompt(question["text"])
        token_rows.append((
            question,
            prompt,
            tokenizer_image_token(
                prompt,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ),
        ))
    return token_rows


def prepare_batch(
    batch,
    questions,
    tokenizer,
    processor,
    model,
    video_folder,
):
    token_rows = build_token_rows(batch, questions, tokenizer)
    per_sample_tensors = []
    for slot, (question, prompt, _tokens) in zip(
        batch["slots"],
        token_rows,
    ):
        seed_all(int(slot["scene_seed"]))
        video = process_videos(
            str(video_folder / slot["scene_id"]),
            processor["video"],
            mode="random",
            device=model.device,
            text=question["text"],
        )
        per_sample_tensors.append({
            key: video[key].to(model.device, dtype=torch.bfloat16)
            for key in ("images", "depths", "poses", "intrinsics")
        })
    batch_tensors = {
        key: torch.cat([sample[key] for sample in per_sample_tensors], dim=0)
        for key in ("images", "depths", "poses", "intrinsics")
    }
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer has neither pad nor EOS token")
    input_ids, attention_mask = left_pad(
        [row[2] for row in token_rows],
        pad_token_id,
        model.device,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "tensors": batch_tensors,
        "questions": [row[0]["question_id"] for row in token_rows],
        "prompts": [row[1] for row in token_rows],
        "tensor_sha256": tensor_dict_sha256(batch_tensors),
    }


def write_json_exclusive(path, payload):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


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
    parser.add_argument("--workload", choices=WORKLOADS, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument(
        "--session-index",
        type=int,
        choices=(1, 2, 3),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()

    for path in (args.out, args.manifest, args.status):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    if args.batch_size < 2:
        raise ValueError("Batch size must be at least two")
    if args.max_new_tokens != 512:
        raise ValueError("Frozen protocol requires max_new_tokens=512")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
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
    full_summary = schedule_summary(schedule)
    if args.max_batches is not None:
        schedule = schedule[:args.max_batches]
    summary = schedule_summary(schedule)
    is_full = (
        full_summary["real_questions"] == FROZEN_QUESTIONS
        and schedule == build_workload_schedule(
            questions,
            inventory,
            args.workload,
            args.batch_size,
            args.seed,
        )
    )
    if not is_full and args.max_batches is None:
        raise ValueError("Full protocol requires all 4675 questions")

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

    root = Path(__file__).resolve().parents[1]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "full mixed-scene heterogeneous GeoGate throughput"
            if is_full else
            "mixed-scene heterogeneous throughput construction"
        ),
        "is_full": is_full,
        "session_index": args.session_index,
        "workload": args.workload,
        "batch_size": args.batch_size,
        "model": {
            "path": str(Path(args.model_path).resolve()),
            "config_sha256": digest_file(
                Path(args.model_path) / "config.json"
            ),
        },
        "questions": {
            "path": str(args.question_file.resolve()),
            "sha256": digest_file(args.question_file),
            "source_count": len(questions),
        },
        "inventory": {
            "path": str(args.inventory.resolve()),
            "sha256": digest_file(args.inventory),
        },
        "protocol": {
            "batch_sizes": [args.batch_size],
            "condition_order": (
                "(batch_index + session_index + workload_index) parity"
            ),
            "schedule_summary": summary,
            "full_schedule_summary": full_summary,
            "warmup_batches": args.warmup_batches,
            "warmup_rows_emitted": False,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "greedy": True,
            "cuda_synchronize": True,
            "timing_boundary": "generate with resident GPU RGB-D tensors",
            "bootstrap_unit": "scene",
            "geogate": {
                "prune_layer": 14,
                "keep_ratio": 0.10,
                "scorer": "last_prompt_attention",
            },
        },
        "provenance": {
            "runner_sha256": digest_file(__file__),
            "schedule_sha256": digest_file(
                root / "analysis/hetero_batch_schedule_v18.py"
            ),
            "batch_patch_sha256": digest_file(
                root / "adaptive_token/geogate_batch_hetero_v18.py"
            ),
            "probe_helpers_sha256": digest_file(
                root / "analysis/run_hetero_batch_v18_probe.py"
            ),
            "audit_sha256": digest_file(
                root / "analysis/audit_hetero_batch_throughput_v18.py"
            ),
        },
    }
    write_json_exclusive(args.manifest, manifest)

    allowed_pids = {os.getpid()}
    measured_batches = 0
    measured_rows = 0
    with args.out.open("x", encoding="utf-8") as output:
        for warmup_index, batch in enumerate(
            schedule[:args.warmup_batches]
        ):
            guard_gpu(
                args.gpu_id,
                allowed_pids,
                f"warmup {warmup_index}",
            )
            prepared = prepare_batch(
                batch,
                questions,
                tokenizer,
                processor,
                baseline,
                args.video_folder,
            )
            for condition in condition_order(
                warmup_index,
                args.session_index,
                0,
            ):
                model = baseline if condition == "baseline" else geogate
                condition_tokenizer = (
                    tokenizer
                    if condition == "baseline"
                    else geogate_tokenizer
                )
                generate_batch(
                    model,
                    condition_tokenizer,
                    prepared["input_ids"],
                    prepared["attention_mask"],
                    prepared["tensors"],
                    args.max_new_tokens,
                )

        for batch in schedule:
            batch_index = batch["batch_index"]
            guard_gpu(
                args.gpu_id,
                allowed_pids,
                f"batch {batch_index}",
            )
            prepared = prepare_batch(
                batch,
                questions,
                tokenizer,
                processor,
                baseline,
                args.video_folder,
            )
            phases = []
            for order, condition in enumerate(condition_order(
                batch_index,
                args.session_index,
                0,
            )):
                model = baseline if condition == "baseline" else geogate
                condition_tokenizer = (
                    tokenizer
                    if condition == "baseline"
                    else geogate_tokenizer
                )
                result = generate_batch(
                    model,
                    condition_tokenizer,
                    prepared["input_ids"],
                    prepared["attention_mask"],
                    prepared["tensors"],
                    args.max_new_tokens,
                )
                phases.append({
                    "condition": condition,
                    "order": order,
                    "result": result,
                    "trace": (
                        list(geogate.model._gg_hetero_trace)
                        if condition == "geogate"
                        else None
                    ),
                })
            if len(phases) != 2:
                raise RuntimeError("Expected both conditions")
            for phase in phases:
                result = phase["result"]
                useful_tokens = sum(
                    len(result["output_ids"][sample_index])
                    for sample_index, slot in enumerate(batch["slots"])
                    if not slot["is_padding"]
                )
                for sample_index, slot in enumerate(batch["slots"]):
                    row = {
                        "session_index": args.session_index,
                        "workload": args.workload,
                        "batch_size": args.batch_size,
                        "batch_index": batch_index,
                        "sample_index": sample_index,
                        "condition": phase["condition"],
                        "order": phase["order"],
                        "question_id": str(slot["question_id"]),
                        "source_question_id": slot["source_question_id"],
                        "question_index": slot["question_index"],
                        "scene_id": slot["scene_id"],
                        "scene_seed": slot["scene_seed"],
                        "visual_tokens": slot["visual_tokens"],
                        "is_padding": slot["is_padding"],
                        "real_questions_in_batch": batch["real_count"],
                        "padding_slots_in_batch": batch["padding_count"],
                        "scene_count_in_batch": batch["scene_count"],
                        "visual_gap_in_batch": batch["visual_gap"],
                        "elapsed_batch_s": result["elapsed_s"],
                        "generated_tokens": len(
                            result["output_ids"][sample_index]
                        ),
                        "useful_generated_tokens_in_batch": useful_tokens,
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
                        "geogate_trace": phase["trace"],
                        "input_tensor_sha256": prepared["tensor_sha256"],
                        "prompt_sha256": (
                            None
                            if slot["is_padding"]
                            else hashlib.sha256(
                                prepared["prompts"][sample_index].encode()
                            ).hexdigest()
                        ),
                    }
                    output.write(json.dumps(row, sort_keys=True) + "\n")
                measured_rows += args.batch_size
                measured_batches += 1
            output.flush()
            if (batch_index + 1) % 25 == 0:
                print(
                    f"{batch_index + 1}/{len(schedule)} batches complete",
                    flush=True,
                )

    expected_rows = 2 * sum(
        batch["batch_size"] for batch in schedule
    )
    expected_batches = 2 * len(schedule)
    if measured_rows != expected_rows or measured_batches != expected_batches:
        raise RuntimeError(
            f"Expected {expected_rows} rows/{expected_batches} batches, "
            f"wrote {measured_rows}/{measured_batches}"
        )
    status = {
        "state": "finished",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "is_full": is_full,
        "session_index": args.session_index,
        "workload": args.workload,
        "batch_size": args.batch_size,
        "batches": len(schedule),
        "rows": measured_rows,
        "raw_sha256": digest_file(args.out),
        "manifest_sha256": digest_file(args.manifest),
    }
    write_json_exclusive(args.status, status)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
