#!/usr/bin/env python3
"""Run the frozen v15 complete-answer intervention on LLaVA-3D."""

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

from adaptive_token.deterministic_scene_cap import (
    audit_scene_cap_trace,
    enable_deterministic_scene_cap,
    set_deterministic_scene_cap,
    tensor_sha256,
)
from adaptive_token.geogate_v0 import enable_geogate
from adaptive_token.prepared_generation_cache_v15 import (
    PreparedGenerationCache,
)
from analysis.readout_generation_v15_protocol import (
    BASELINE,
    FROZEN_SEED,
    MAX_NEW_TOKENS,
    arm_by_name,
    balanced_scene_shards,
    condition_names,
    delete_count,
)
from analysis.readout_generation_v15_trace import (
    GenerationTrace,
    summarize_trace,
)
from analysis.review_v3_model_provenance import provenance


FROZEN_SCENES = 71
FROZEN_QUESTIONS = 4675


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def question_arm_seed(question_id, arm_seed):
    payload = f"3dvlm-v15-generation\0{question_id}\0{arm_seed}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (
        2 ** 63 - 1
    )


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


def selected_questions(questions, inventory, only_scenes, per_scene):
    scene_order = [row["scene_id"] for row in inventory["scene_rows"]]
    scene_info = {
        row["scene_id"]: row
        for row in inventory["scene_rows"]
    }
    if only_scenes:
        aliases = {
            scene_id.rsplit("/", 1)[-1]: scene_id
            for scene_id in scene_order
        }
        requested = []
        for value in only_scenes:
            normalized = aliases.get(value, value)
            if normalized not in scene_info:
                raise ValueError(f"Unknown inventory scene: {value}")
            if normalized not in requested:
                requested.append(normalized)
        scene_order = requested
    by_scene = {scene_id: [] for scene_id in scene_order}
    for row in questions:
        if row["video"] in by_scene:
            by_scene[row["video"]].append(row)
    selected = []
    for scene_id in scene_order:
        rows = by_scene[scene_id]
        if not rows:
            raise ValueError(f"No questions found for {scene_id}")
        selected.extend(rows[:per_scene] if per_scene else rows)
    return selected, scene_info


def configure_baseline(model, native_forward):
    model.model.forward = native_forward
    model.model._gg_span = None
    model.model._gg_next_pos = None
    model.model._gg_stages = {}
    model.model._gg_attn_history = []


def configure_arm(model, geogate_forward, arm, spatial_start, spatial_tokens):
    lm = model.model
    lm.forward = geogate_forward
    lm._gg_prune_layer = arm["layer"]
    lm._gg_stages = {arm["layer"]: 1.0}
    lm._gg_stages_user_set = True
    lm._gg_scorer = arm["scorer"]
    lm._gg_prune_mode = arm["prune_mode"]
    lm._gg_delete_n = delete_count(spatial_tokens)
    lm._gg_span = (spatial_start, spatial_start + spatial_tokens)
    lm._gg_next_pos = None
    lm._gg_attn_history = []
    lm._gg_coords = None


def generate(model, tokenizer, input_ids, tensors, trace):
    trace.reset()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=tensors["images"],
            depths=tensors["depths"],
            poses=tensors["poses"],
            intrinsics=tensors["intrinsics"],
            image_sizes=None,
            do_sample=False,
            num_beams=1,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
        )
    torch.cuda.synchronize()
    output_cpu = output_ids.detach().cpu()
    return {
        "answer": tokenizer.batch_decode(
            output_cpu,
            skip_special_tokens=True,
        )[0].strip(),
        "output_ids": output_cpu[0].tolist(),
        "output_ids_sha256": tensor_sha256(output_cpu),
        "latency_s": time.perf_counter() - start,
    }


def run_condition(
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
    question_id,
):
    if condition == BASELINE:
        configure_baseline(model, native_forward)
        arm = None
        seed = 0
        reduced = full_tokens
    else:
        arm = arm_by_name()[condition]
        configure_arm(
            model,
            geogate_forward,
            arm,
            spatial_start,
            spatial_tokens,
        )
        seed = arm["seed"]
        reduced = full_tokens - delete_count(spatial_tokens)
    seed_all(question_arm_seed(question_id, seed))
    result = generate(model, tokenizer, input_ids, tensors, trace)
    selected = (
        list(range(full_tokens))
        if condition == BASELINE
        else model.model._gg_orig_map.detach().cpu().tolist()
    )
    trace_summary = summarize_trace(
        trace,
        full_tokens,
        None if arm is None else arm["layer"],
        None if arm is None else reduced,
    )
    if tensor_sha256(prepared[4]) != expanded_hash:
        raise RuntimeError("Prepared embedding changed during generation")
    return {
        **result,
        **trace_summary,
        "selected_original_indices": selected,
        "actual_deleted_spatial_tokens": (
            0 if arm is None else delete_count(spatial_tokens)
        ),
        "layer": None if arm is None else arm["layer"],
        "ranking": None if arm is None else arm["ranking"],
        "seed": seed,
        "forward_path": (
            "native_unpatched_llama_forward"
            if arm is None else
            "geogate_intervention_forward"
        ),
    }


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
    question_id,
    records,
):
    expected = {row["condition"]: row for row in records}
    for condition in condition_names():
        repeated = run_condition(
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
            raise RuntimeError(f"Repeated v15 condition changed: {condition}")


def write_json_exclusive(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--question-file", type=Path, required=True)
    parser.add_argument("--video-folder", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--only-scene", action="append")
    parser.add_argument("--questions-per-scene", type=int)
    parser.add_argument("--repeat-verification", action="store_true")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int)
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
    guard_gpu(args.gpu_id, set(), "preflight")

    questions = json.loads(args.question_file.read_text(encoding="utf-8"))
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    if (
        len(questions) != FROZEN_QUESTIONS
        or len({str(row["question_id"]) for row in questions})
        != FROZEN_QUESTIONS
        or inventory["questions"]["sha256"] != digest_file(args.question_file)
    ):
        raise ValueError("Question file does not match the frozen inventory")
    if (args.shard_index is None) != (args.num_shards is None):
        raise ValueError("Set both --shard-index and --num-shards")
    if args.only_scene and args.shard_index is not None:
        raise ValueError("--only-scene and sharding are mutually exclusive")
    only_scenes = args.only_scene
    shard_load = None
    if args.shard_index is not None:
        shards, loads = balanced_scene_shards(inventory, args.num_shards)
        if not 0 <= args.shard_index < len(shards):
            raise ValueError("Shard index outside configured range")
        only_scenes = shards[args.shard_index]
        shard_load = loads[args.shard_index]
    selected, scene_info = selected_questions(
        questions,
        inventory,
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
        keep_ratio=0.90,
        scorer="attn",
        prune_mode="delete_topk",
        delete_n=1,
    )
    geogate_forward = model.model.forward
    enable_deterministic_scene_cap(model)
    cache = PreparedGenerationCache(model)
    trace = GenerationTrace(model)

    root = Path(__file__).resolve().parents[1]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "full frozen v15 answer intervention"
            if is_full else "v15 construction or shard smoke"
        ),
        "is_full": is_full,
        "question_ids": [str(row["question_id"]) for row in selected],
        "scenes": list(dict.fromkeys(row["video"] for row in selected)),
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
            "selected_count": len(selected),
        },
        "inventory": {
            "path": str(args.inventory.resolve()),
            "sha256": digest_file(args.inventory),
        },
        "protocol": {
            "conditions": condition_names(),
            "layers": [8, 14, 20],
            "delete_fraction": 0.10,
            "random_arm_seeds": [101, 202, 303],
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
            "bootstrap": {
                "unit": "scene",
                "iterations": 10000,
                "seed": FROZEN_SEED,
                "primary_interval": 0.975,
            },
            "repeat_verification": bool(args.repeat_verification),
        },
        "shard": {
            "index": args.shard_index,
            "count": args.num_shards,
            "estimated_attention_weight": shard_load,
        },
        "provenance": {
            "runner_sha256": digest_file(__file__),
            "protocol_sha256": digest_file(
                root / "analysis/readout_generation_v15_protocol.py"
            ),
            "trace_sha256": digest_file(
                root / "analysis/readout_generation_v15_trace.py"
            ),
            "cache_sha256": digest_file(
                root / "adaptive_token/prepared_generation_cache_v15.py"
            ),
            "audit_sha256": digest_file(
                root / "analysis/audit_readout_generation_v15.py"
            ),
            "scorer_sha256": digest_file(
                root / "analysis/score_readout_generation_v15.py"
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
                    result = run_condition(
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
                        question_id,
                    )
                    records.append({
                        **common,
                        **result,
                        "condition": condition,
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
                        question_id,
                        records,
                    )
                    if cache.prefill_hits != 2 * len(condition_names()):
                        raise RuntimeError("Repeat did not reuse every condition")
                for record in records:
                    record["repeat_verified"] = bool(
                        args.repeat_verification
                    )
                    record["final_prepared_prefill_cache_hits"] = (
                        cache.prefill_hits
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
    }
    write_json_exclusive(args.status, status)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
