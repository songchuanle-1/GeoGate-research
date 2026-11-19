#!/usr/bin/env python3
"""Run same-token-set deletion at different layers on LLaVA-3D."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import torch

from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import (
    get_model_name_from_path,
    process_videos,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model

from adaptive_token.deterministic_scene_cap import (
    audit_scene_cap_trace,
    enable_deterministic_scene_cap,
    set_deterministic_scene_cap,
    tensor_sha256,
)
from adaptive_token.geogate_v0_fixedset_v21 import (
    enable_geogate,
    set_fixed_spatial_deletion,
)
from adaptive_token.prepared_generation_cache_v15 import PreparedGenerationCache
from analysis.readout_fixed_deletion_protocol import (
    BASELINE,
    FROZEN_SEED,
    arm_by_name,
    condition_names,
    digest_indices,
    expected_selected,
)
from analysis.readout_generation_v15_protocol import (
    MAX_NEW_TOKENS,
    balanced_scene_shards,
)
from analysis.readout_generation_v15_trace import (
    GenerationTrace,
    summarize_trace,
)
from analysis.experiment_lock import lock_reference, require_lock
from analysis.review_v3_model_provenance import provenance
from analysis.run_readout_generation_v15 import (
    build_prompt,
    configure_arm,
    configure_baseline,
    digest_file,
    generate,
    guard_gpu,
    question_arm_seed,
    seed_all,
    selected_questions,
    write_json_exclusive,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN_SCENES = 71
FROZEN_QUESTIONS = 4675
DEFAULT_FIXED_MANIFEST = (
    ROOT / "results/raw/followup/fixed_deletion_v21_manifest_20260917.json"
)
DEFAULT_LOCK = (
    ROOT / "results/raw/followup/fixed_deletion_v21_experiment_lock_v2.json"
)


def load_fixed_manifest(path: Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema") != "geogate-fixed-deletion-manifest-v1":
        raise ValueError("Unexpected fixed-deletion manifest schema")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != FROZEN_QUESTIONS:
        raise ValueError("Expected 4,675 fixed-deletion entries")
    by_id = {}
    for entry in entries:
        question_id = str(entry["question_id"])
        if question_id in by_id:
            raise ValueError(f"Duplicate fixed-deletion entry: {question_id}")
        by_id[question_id] = entry
    return by_id


def manifest_projection(manifest_data):
    return {
        key: manifest_data[key]
        for key in (
            "is_full",
            "question_ids",
            "scenes",
            "model",
            "questions",
            "inventory",
            "protocol",
        )
    }


def fixed_manifest_hashes(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "path": str(Path(path).resolve()),
        "sha256": digest_file(path),
        "source_hashes": data["source_hashes"],
        "questions": data["questions"],
    }


def select_questions_with_fixed_entries(
    questions,
    inventory,
    fixed_entries,
    only_scenes,
    per_scene,
):
    selected, scene_info = selected_questions(
        questions,
        inventory,
        only_scenes,
        per_scene,
    )
    missing = [
        str(row["question_id"])
        for row in selected
        if str(row["question_id"]) not in fixed_entries
    ]
    if missing:
        raise ValueError(f"Missing fixed deletion entries: {missing[:3]}")
    return selected, scene_info


def verify_repeat(
    model,
    tokenizer,
    input_ids,
    tensors,
    trace,
    prepared,
    expanded_hash,
    full_tokens,
    spatial_start,
    spatial_tokens,
    native_forward,
    geogate_forward,
    fixed_entry,
    question_id,
    records,
):
    expected = {row["condition"]: row for row in records}
    for condition in condition_names():
        if condition == BASELINE:
            deleted = []
        else:
            arm = arm_by_name()[condition]
            source = f"L{int(arm['source_layer'])}"
            deleted = fixed_entry["source_sets"][source][
                "deleted_original_indices"
            ]
        repeated = run_fixed_condition(
            model,
            tokenizer,
            input_ids,
            tensors,
            trace,
            prepared,
            expanded_hash,
            full_tokens,
            spatial_start,
            spatial_tokens,
            condition,
            native_forward,
            geogate_forward,
            deleted,
            question_id,
        )
        row = expected[condition]
        if (
            repeated["output_ids_sha256"] != row["output_ids_sha256"]
            or repeated["selected_original_indices"]
            != row["selected_original_indices"]
            or repeated["complete_trace_sha256"]
            != row["complete_trace_sha256"]
        ):
            raise RuntimeError(
                f"Repeated fixed-deletion condition changed: {condition}"
            )


def configure_fixed_deletion(
    model,
    geogate_forward,
    prune_layer,
    spatial_start,
    spatial_tokens,
    deleted_indices,
):
    configure_arm(
        model,
        geogate_forward,
        {
            "layer": int(prune_layer),
            "scorer": "attn",
            "prune_mode": "delete_topk",
        },
        spatial_start,
        spatial_tokens,
    )
    set_fixed_spatial_deletion(model, deleted_indices)


def run_fixed_condition(
    model,
    tokenizer,
    input_ids,
    tensors,
    trace,
    prepared,
    expanded_hash,
    full_tokens,
    spatial_start,
    spatial_tokens,
    condition,
    native_forward,
    geogate_forward,
    deleted_indices,
    question_id,
):
    if condition == BASELINE:
        configure_baseline(model, native_forward)
        prune_layer = None
        source_layer = None
        deleted_indices = []
        seed = 0
        reduced = full_tokens
    else:
        arm = arm_by_name()[condition]
        prune_layer = int(arm["prune_layer"])
        source_layer = int(arm["source_layer"])
        configure_fixed_deletion(
            model,
            geogate_forward,
            prune_layer,
            spatial_start,
            spatial_tokens,
            deleted_indices,
        )
        seed = FROZEN_SEED
        reduced = full_tokens - len(deleted_indices)

    seed_all(question_arm_seed(question_id, seed))
    result = generate(model, tokenizer, input_ids, tensors, trace)
    selected = (
        list(range(full_tokens))
        if condition == BASELINE
        else model.model._gg_orig_map.detach().cpu().tolist()
    )
    expected = expected_selected(full_tokens, deleted_indices)
    if selected != expected:
        raise RuntimeError(
            f"{condition}: fixed deletion selection changed"
        )
    if tensor_sha256(prepared[4]) != expanded_hash:
        raise RuntimeError("Prepared embedding changed during generation")
    trace_summary = summarize_trace(
        trace,
        full_tokens,
        prune_layer,
        None if prune_layer is None else reduced,
    )
    return {
        **result,
        **trace_summary,
        "condition": condition,
        "layer": prune_layer,
        "ranking": (
            None
            if source_layer is None
            else f"fixed_top_from_L{source_layer}"
        ),
        "source_layer": source_layer,
        "seed": seed,
        "selected_original_indices": selected,
        "actual_deleted_spatial_tokens": len(deleted_indices),
        "fixed_deleted_original_indices": deleted_indices,
        "fixed_deleted_sha256": (
            None
            if not deleted_indices
            else digest_indices(deleted_indices)
        ),
        "forward_path": (
            "native_unpatched_llama_forward"
            if condition == BASELINE else
            "geogate_fixed_deletion_forward"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--question-file", type=Path, required=True)
    parser.add_argument("--video-folder", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument(
        "--fixed-manifest",
        type=Path,
        default=DEFAULT_FIXED_MANIFEST,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--only-scene", action="append")
    parser.add_argument("--questions-per-scene", type=int)
    parser.add_argument("--repeat-verification", action="store_true")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()

    for path in (args.out, args.manifest, args.status):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    if (
        args.questions_per_scene is not None
        and args.questions_per_scene < 1
    ):
        raise ValueError("--questions-per-scene must be positive")
    if (args.shard_index is None) != (args.num_shards is None):
        raise ValueError("Set both --shard-index and --num-shards")
    if args.only_scene and args.shard_index is not None:
        raise ValueError("--only-scene and sharding are mutually exclusive")
    lock = lock_reference(require_lock(args.lock, ROOT, "fixed_deletion_v21"))
    metadata = lock["metadata"]
    if (
        int(metadata.get("questions", -1)) != FROZEN_QUESTIONS
        or int(metadata.get("scenes", -1)) != FROZEN_SCENES
        or int(metadata.get("seed", -1)) != FROZEN_SEED
        or list(metadata.get("source_layers", [])) != [8, 14, 20]
        or list(metadata.get("prune_layers", [])) != [8, 14, 20]
    ):
        raise RuntimeError("Fixed-deletion experiment lock metadata changed")
    guard_gpu(args.gpu_id, set(), "preflight")

    questions = json.loads(args.question_file.read_text(encoding="utf-8"))
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    fixed_entries = load_fixed_manifest(args.fixed_manifest)
    if (
        len(questions) != FROZEN_QUESTIONS
        or len({str(row["question_id"]) for row in questions})
        != FROZEN_QUESTIONS
        or inventory["questions"]["sha256"] != digest_file(args.question_file)
    ):
        raise ValueError("Question file does not match the frozen inventory")
    only_scenes = args.only_scene
    shard_load = None
    if args.shard_index is not None:
        shards, loads = balanced_scene_shards(inventory, args.num_shards)
        if not 0 <= args.shard_index < len(shards):
            raise ValueError("Shard index outside configured range")
        only_scenes = shards[args.shard_index]
        shard_load = loads[args.shard_index]
    selected, scene_info = select_questions_with_fixed_entries(
        questions,
        inventory,
        fixed_entries,
        only_scenes,
        args.questions_per_scene,
    )
    scene_count = len({row["video"] for row in selected})
    is_full = (
        len(selected) == FROZEN_QUESTIONS
        and scene_count == FROZEN_SCENES
    )
    if not args.only_scene and args.shard_index is None and not is_full:
        raise ValueError("Non-smoke execution must contain the full set")

    seed_all(FROZEN_SEED)
    torch.use_deterministic_algorithms(True)
    tokenizer, model, processor, _ = load_pretrained_model(
        args.model_path,
        None,
        get_model_name_from_path(args.model_path),
        device_map="cuda:0",
    )
    model.eval()
    native_forward = model.model.forward
    enable_geogate(
        model,
        prune_layer=14,
        keep_ratio=0.10,
        scorer="attn",
        prune_mode="delete_topk",
        delete_n=1,
    )
    geogate_forward = model.model.forward
    enable_deterministic_scene_cap(model)
    cache = PreparedGenerationCache(model)
    trace = GenerationTrace(model)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "full fixed-token-set cross-layer deletion"
            if is_full else "fixed-deletion construction smoke"
        ),
        "is_full": is_full,
        "question_ids": [str(row["question_id"]) for row in selected],
        "scenes": list(dict.fromkeys(row["video"] for row in selected)),
        "model": {
            "path": str(Path(args.model_path).resolve()),
            "config_sha256": digest_file(Path(args.model_path) / "config.json"),
        },
        "questions": {
            "path": str(args.question_file.resolve()),
            "sha256": digest_file(args.question_file),
            "source_count": len(questions),
            "selected_count": len(selected),
        },
        "inventory": {
            "path": str(args.inventory.resolve()),
            "sha256": digest_file(args.inventory),
        },
        "fixed_deletion_manifest": fixed_manifest_hashes(
            args.fixed_manifest
        ),
        "experiment_lock": lock,
        "protocol": {
            "conditions": condition_names(),
            "source_layers": [8, 14, 20],
            "prune_layers": [8, 14, 20],
            "seed": FROZEN_SEED,
            "prompt": "llava_v1: <image>\\n{question}",
            "decoding": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": MAX_NEW_TOKENS,
                "use_cache": True,
            },
            "prepared_input": (
                "one exact prepared embedding object reused by all conditions"
            ),
            "repeat_verification": bool(args.repeat_verification),
        },
        "shard": {
            "index": args.shard_index,
            "count": args.num_shards,
            "estimated_attention_weight": shard_load,
        },
        "provenance": {
            "runner_sha256": digest_file(__file__),
            "fixed_protocol_sha256": digest_file(
                ROOT / "analysis/readout_fixed_deletion_protocol.py"
            ),
            "v15_runner_sha256": digest_file(
                ROOT / "analysis/run_readout_generation_v15.py"
            ),
            "v15_trace_sha256": digest_file(
                ROOT / "analysis/readout_generation_v15_trace.py"
            ),
            "cache_sha256": digest_file(
                ROOT / "adaptive_token/prepared_generation_cache_v15.py"
            ),
            "geogate_sha256": digest_file(inspect.getfile(enable_geogate)),
            "cap_sha256": digest_file(
                inspect.getfile(enable_deterministic_scene_cap)
            ),
            "model_and_sources": provenance(
                model,
                args.model_path,
                cache.original,
            ),
        },
    }
    write_json_exclusive(args.manifest, manifest)

    allowed_pids = {os.getpid()}
    completed = 0
    try:
        with args.out.open("x", encoding="utf-8") as handle:
            for index, row in enumerate(selected):
                guard_gpu(args.gpu_id, allowed_pids, f"question {index}")
                question_id = str(row["question_id"])
                scene_id = row["video"]
                scene_seed = int(scene_info[scene_id]["scene_seed"])
                prompt = build_prompt(row["text"])
                input_ids = tokenizer_image_token(
                    prompt,
                    tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                ).unsqueeze(0).to(model.device)
                positions = (
                    input_ids[0] == IMAGE_TOKEN_INDEX
                ).nonzero(as_tuple=False).flatten()
                if positions.numel() != 1:
                    raise RuntimeError("Expected exactly one image placeholder")
                spatial_start = int(positions[0].item())

                seed_all(scene_seed)
                videos = process_videos(
                    str(args.video_folder / scene_id),
                    processor["video"],
                    mode="random",
                    device=model.device,
                    text=row["text"],
                )
                tensors = {
                    key: videos[key].to(model.device, dtype=torch.bfloat16)
                    for key in ("images", "depths", "poses", "intrinsics")
                }
                input_hashes = {
                    key: tensor_sha256(value)
                    for key, value in tensors.items()
                }
                cache.clear()
                set_deterministic_scene_cap(model, scene_id, scene_seed)
                prepare_start = time.perf_counter()
                with torch.inference_mode():
                    prepared = cache.prepare(
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
                torch.cuda.synchronize()
                preparation_latency_s = time.perf_counter() - prepare_start
                full_tokens = int(prepared[4].shape[1])
                spatial_tokens = full_tokens - int(input_ids.shape[1]) + 1
                if spatial_tokens != int(scene_info[scene_id]["visual_tokens"]):
                    raise RuntimeError("Prepared visual count differs from inventory")
                cap_trace = list(model._deterministic_scene_cap_trace)
                pre_cap_tokens = (
                    cap_trace[0]["token_count_before_cap"]
                    if cap_trace else spatial_tokens
                )
                cap_summary = audit_scene_cap_trace(model, 1, pre_cap_tokens)
                expanded_hash = tensor_sha256(prepared[4])
                fixed_entry = fixed_entries[question_id]
                if fixed_entry["expanded_input_sha256"] != expanded_hash:
                    raise RuntimeError(
                        f"{question_id}: fixed manifest input hash mismatch"
                    )
                if (
                    int(fixed_entry["full_tokens"]) != full_tokens
                    or int(fixed_entry["spatial_start"]) != spatial_start
                    or int(fixed_entry["input_visual_tokens"])
                    != spatial_tokens
                ):
                    raise RuntimeError(
                        f"{question_id}: fixed manifest span mismatch"
                    )
                deleted_by_source = {
                    int(source[1:]): payload["deleted_original_indices"]
                    for source, payload in fixed_entry["source_sets"].items()
                }
                common = {
                    "question_id": question_id,
                    "scene_id": scene_id,
                    "prompt": prompt,
                    "scene_seed": scene_seed,
                    "input_tensor_hashes": input_hashes,
                    "expanded_input_sha256": expanded_hash,
                    "full_tokens": full_tokens,
                    "spatial_start": spatial_start,
                    "input_visual_tokens": spatial_tokens,
                    "deterministic_cap": cap_summary,
                    "preparation_latency_s": preparation_latency_s,
                    "preparation_calls": 1,
                    "condition_count": len(condition_names()),
                    "max_new_tokens": MAX_NEW_TOKENS,
                }
                records = []
                for order, condition in enumerate(condition_names()):
                    if condition == BASELINE:
                        deleted = []
                    else:
                        arm = arm_by_name()[condition]
                        deleted = deleted_by_source[int(arm["source_layer"])]
                    result = run_fixed_condition(
                        model,
                        tokenizer,
                        input_ids,
                        tensors,
                        trace,
                        prepared,
                        expanded_hash,
                        full_tokens,
                        spatial_start,
                        spatial_tokens,
                        condition,
                        native_forward,
                        geogate_forward,
                        deleted,
                        question_id,
                    )
                    records.append({
                        **common,
                        **result,
                        "condition_order": order,
                        "prepared_prefill_cache_hit": order + 1,
                        "prepared_embedding_verified_after_forward": True,
                    })
                if cache.prefill_hits != len(condition_names()):
                    raise RuntimeError("Not every condition reused the preparation")
                if args.repeat_verification:
                    verify_repeat(
                        model,
                        tokenizer,
                        input_ids,
                        tensors,
                        trace,
                        prepared,
                        expanded_hash,
                        full_tokens,
                        spatial_start,
                        spatial_tokens,
                        native_forward,
                        geogate_forward,
                        fixed_entry,
                        question_id,
                        records,
                    )
                    if cache.prefill_hits != 2 * len(condition_names()):
                        raise RuntimeError(
                            "Repeat did not reuse every condition"
                        )
                for record in records:
                    record["final_prepared_prefill_cache_hits"] = (
                        cache.prefill_hits
                    )
                    record["repeat_verified"] = bool(
                        args.repeat_verification
                    )
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                completed += 1
                cache.clear()
                del prepared, records, tensors, videos
                torch.cuda.empty_cache()
                print(
                    f"{completed}/{len(selected)} complete: "
                    f"{question_id} ({scene_id})",
                    flush=True,
                )
    finally:
        trace.close()
        cache.close()

    status = {
        "state": "finished",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "is_full": is_full,
        "questions": completed,
        "scenes": scene_count,
        "conditions": len(condition_names()),
        "rows": completed * len(condition_names()),
        "repeat_verification": bool(args.repeat_verification),
        "raw_sha256": digest_file(args.out),
        "manifest_sha256": digest_file(args.manifest),
        "experiment_lock": lock,
    }
    write_json_exclusive(args.status, status)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
