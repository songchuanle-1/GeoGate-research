#!/usr/bin/env python3
"""Run the frozen v13 first-token readout intervention on LLaVA-3D."""

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
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
from adaptive_token.geogate_v0 import enable_auto_span, enable_geogate
from analysis.readout_logit_v13_protocol import (
    BASELINE,
    arm_by_name,
    balanced_scene_shards,
    condition_names,
    delete_count,
)
from analysis.review_v3_model_provenance import provenance


FROZEN_SEED = 20260913
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
    payload = f"3dvlm-v13-random\0{question_id}\0{arm_seed}".encode()
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
    inventory_rows = inventory["scene_rows"]
    scene_order = [row["scene_id"] for row in inventory_rows]
    scene_info = {row["scene_id"]: row for row in inventory_rows}
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
        scene_id = row["video"]
        if scene_id in by_scene:
            by_scene[scene_id].append(row)
    selected = []
    for scene_id in scene_order:
        rows = by_scene[scene_id]
        if not rows:
            raise ValueError(f"No questions found for {scene_id}")
        selected.extend(rows[:per_scene] if per_scene else rows)
    return selected, scene_info


class LayerLengthTrace:
    def __init__(self, model):
        self.lengths = []
        self.handles = []
        for index, layer in enumerate(model.model.layers):
            def before(module, args, kwargs, index=index):
                hidden = args[0] if args else kwargs["hidden_states"]
                self.lengths.append((index, int(hidden.shape[1])))
            self.handles.append(
                layer.register_forward_pre_hook(before, with_kwargs=True)
            )

    def reset(self):
        self.lengths.clear()

    def values(self):
        if len(self.lengths) != 32:
            raise RuntimeError(
                f"Expected 32 layer calls, observed {len(self.lengths)}"
            )
        if [index for index, _ in self.lengths] != list(range(32)):
            raise RuntimeError("Language-model layers executed out of order")
        return [length for _, length in self.lengths]


def target_margin(logits, target):
    target_logit = logits[target]
    masked = logits.clone()
    masked[target] = -torch.inf
    return float((target_logit - masked.max()).item())


def js_divergence(baseline_log_probs, intervention_log_probs):
    baseline_probs = baseline_log_probs.exp()
    intervention_probs = intervention_log_probs.exp()
    mixture = 0.5 * (baseline_probs + intervention_probs)
    log_mixture = mixture.clamp_min(torch.finfo(mixture.dtype).tiny).log()
    value = 0.5 * (
        (baseline_probs * (baseline_log_probs - log_mixture)).sum()
        + (
            intervention_probs
            * (intervention_log_probs - log_mixture)
        ).sum()
    )
    return float(value.item())


def forward_last(model, prepared, trace, forward_fn):
    trace.reset()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = forward_fn(
            input_ids=None,
            inputs_embeds=prepared[4],
            position_ids=prepared[1],
            attention_mask=prepared[2],
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        final_hidden = output.last_hidden_state[:, -1, :]
        logits = model.lm_head(final_hidden).float()[0]
    torch.cuda.synchronize()
    return {
        "hidden": final_hidden.float()[0],
        "logits": logits,
        "log_probs": torch.log_softmax(logits, dim=-1),
        "latency_s": time.perf_counter() - start,
        "layer_lengths": trace.values(),
    }


def configure_baseline(model):
    model.model._gg_span = None
    model.model._gg_stages = {}
    model.model._gg_attn_history = []


def configure_arm(model, arm, spatial_start, spatial_tokens):
    lm = model.model
    lm._gg_prune_layer = arm["layer"]
    lm._gg_stages = {arm["layer"]: 1.0}
    lm._gg_stages_user_set = True
    lm._gg_scorer = arm["scorer"]
    lm._gg_prune_mode = arm["prune_mode"]
    lm._gg_delete_n = delete_count(spatial_tokens)
    lm._gg_span = (spatial_start, spatial_start + spatial_tokens)
    lm._gg_attn_history = []
    lm._gg_coords = None


def manifest_payload(args, questions, inventory, selected, model, prepare):
    if args.shard_index is not None:
        scope = "confirmatory v13 full-run shard"
    elif len(selected) == FROZEN_QUESTIONS:
        scope = "full confirmatory v13 readout-logit intervention"
    else:
        scope = "v13 construction and maximum-token scene smoke"
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "is_full": len(selected) == FROZEN_QUESTIONS,
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
            "random_question_seed": (
                "sha256('3dvlm-v13-random\\0' + question_id + "
                "'\\0' + arm_seed), first 8 bytes modulo 2^63-1"
            ),
            "seed": FROZEN_SEED,
            "scene_seed": "fixed values from v7 inventory",
            "numeric_mode": "bf16 model, float32 endpoint computation",
            "use_cache": False,
            "decode_forward_calls": 0,
            "repeat_verification": bool(args.repeat_verification),
            "primary_endpoint": (
                "baseline log p(y_base) minus intervention log p(y_base)"
            ),
            "bootstrap": {
                "unit": "scene",
                "iterations": 10000,
                "seed": FROZEN_SEED,
                "primary_interval": 0.975,
            },
        },
        "provenance": {
            "runner_sha256": digest_file(__file__),
            "protocol_sha256": digest_file(
                Path(__file__).with_name("readout_logit_v13_protocol.py")
            ),
            "audit_sha256": digest_file(
                Path(__file__).with_name("audit_readout_logit_v13.py")
            ),
            "scorer_sha256": digest_file(
                Path(__file__).with_name("score_readout_logit_v13.py")
            ),
            "geogate_sha256": digest_file(
                inspect.getfile(enable_geogate)
            ),
            "cap_sha256": digest_file(
                inspect.getfile(enable_deterministic_scene_cap)
            ),
            "model_and_sources": provenance(
                model,
                args.model_path,
                prepare,
            ),
        },
    }


def write_json_exclusive(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def verify_repeat(
    model,
    prepared,
    trace,
    native_forward,
    geogate_forward,
    question_id,
    records,
):
    expected = {row["condition"]: row for row in records}
    arms = arm_by_name()
    configure_baseline(model)
    seed_all(question_arm_seed(question_id, 0))
    baseline = forward_last(model, prepared, trace, native_forward)
    if (
        tensor_sha256(baseline["log_probs"])
        != expected[BASELINE]["logit_distribution_sha256"]
        or baseline["layer_lengths"]
        != expected[BASELINE]["prefill_layer_input_lengths"]
    ):
        raise RuntimeError("Repeated baseline prefill changed")
    for condition in condition_names()[1:]:
        row = expected[condition]
        arm = arms[condition]
        configure_arm(
            model,
            arm,
            int(row["spatial_start"]),
            int(row["input_visual_tokens"]),
        )
        seed_all(question_arm_seed(question_id, arm["seed"]))
        intervention = forward_last(
            model,
            prepared,
            trace,
            geogate_forward,
        )
        selected = model.model._gg_orig_map.detach().cpu().tolist()
        if (
            tensor_sha256(intervention["log_probs"])
            != row["logit_distribution_sha256"]
            or selected != row["selected_original_indices"]
            or intervention["layer_lengths"]
            != row["prefill_layer_input_lengths"]
        ):
            raise RuntimeError(f"Repeated {condition} prefill changed")


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

    seed_all(FROZEN_SEED)
    torch.use_deterministic_algorithms(True)
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
        if not 0 <= args.shard_index < args.num_shards:
            raise ValueError("Shard index outside configured range")
        shards, loads = balanced_scene_shards(
            inventory,
            args.num_shards,
        )
        only_scenes = shards[args.shard_index]
        shard_load = loads[args.shard_index]
    selected_question_rows, scene_info = selected_questions(
        questions,
        inventory,
        only_scenes,
        args.questions_per_scene,
    )
    selected_scene_count = len({
        row["video"] for row in selected_question_rows
    })
    is_full = (
        len(selected_question_rows) == FROZEN_QUESTIONS
        and selected_scene_count == FROZEN_SCENES
    )
    if (
        not args.only_scene
        and args.shard_index is None
        and not is_full
    ):
        raise ValueError("Non-smoke execution must contain the full test set")

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
    enable_auto_span(model, IMAGE_TOKEN_INDEX)
    original_prepare = model.prepare_inputs_labels_for_multimodal
    trace = LayerLengthTrace(model)
    manifest = manifest_payload(
        args,
        questions,
        inventory,
        selected_question_rows,
        model,
        original_prepare,
    )
    manifest["shard"] = {
        "index": args.shard_index,
        "count": args.num_shards,
        "estimated_attention_weight": shard_load,
    }
    write_json_exclusive(args.manifest, manifest)

    allowed_pids = {os.getpid()}
    completed = 0
    arms = arm_by_name()
    with args.out.open("x", encoding="utf-8") as handle:
        for question_index, row in enumerate(selected_question_rows):
            guard_gpu(
                args.gpu_id,
                allowed_pids,
                f"question {question_index}",
            )
            question_id = str(row["question_id"])
            scene_id = row["video"]
            info = scene_info[scene_id]
            scene_seed = int(info["scene_seed"])
            prompt = build_prompt(row["text"])
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
                raise RuntimeError("Expected exactly one image placeholder")
            spatial_start = int(image_positions[0].item())

            seed_all(scene_seed)
            videos = process_videos(
                str(args.video_folder / scene_id),
                processor["video"],
                mode="random",
                device=model.device,
                text=row["text"],
            )
            tensors = {
                key: videos[key].to(
                    model.device,
                    dtype=torch.bfloat16,
                )
                for key in ("images", "depths", "poses", "intrinsics")
            }
            input_hashes = {
                key: tensor_sha256(value)
                for key, value in tensors.items()
            }
            set_deterministic_scene_cap(model, scene_id, scene_seed)
            prepare_start = time.perf_counter()
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
                    None,
                    None,
                )
            torch.cuda.synchronize()
            preparation_latency_s = time.perf_counter() - prepare_start
            if prepared[4] is None or prepared[4].shape[0] != 1:
                raise RuntimeError("Invalid prepared embedding")
            full_tokens = int(prepared[4].shape[1])
            spatial_tokens = full_tokens - int(input_ids.shape[1]) + 1
            if spatial_tokens != int(info["visual_tokens"]):
                raise RuntimeError(
                    f"{scene_id}: prepared {spatial_tokens} visual tokens, "
                    f"inventory records {info['visual_tokens']}"
                )
            cap_trace = list(model._deterministic_scene_cap_trace)
            pre_cap_tokens = (
                cap_trace[0]["token_count_before_cap"]
                if cap_trace else spatial_tokens
            )
            cap = audit_scene_cap_trace(model, 1, pre_cap_tokens)
            expanded_hash = tensor_sha256(prepared[4])

            configure_baseline(model)
            seed_all(question_arm_seed(question_id, 0))
            baseline = forward_last(
                model,
                prepared,
                trace,
                native_forward,
            )
            baseline_token = int(torch.argmax(baseline["logits"]).item())
            baseline_logprob = float(
                baseline["log_probs"][baseline_token].item()
            )
            baseline_margin = target_margin(
                baseline["logits"],
                baseline_token,
            )
            baseline_distribution_hash = tensor_sha256(
                baseline["log_probs"]
            )
            baseline_selected = list(range(full_tokens))
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
                "deterministic_cap": cap,
                "baseline_token_id": baseline_token,
                "baseline_distribution_sha256": (
                    baseline_distribution_hash
                ),
                "baseline_token_logprob": baseline_logprob,
                "baseline_token_margin": baseline_margin,
                "preparation_latency_s": preparation_latency_s,
                "last_prompt_original_index": full_tokens - 1,
                "final_output_original_index": full_tokens - 1,
                "prepared_embedding_verified_after_forward": True,
                "use_cache": False,
                "decode_forward_calls": 0,
                "baseline_forward_path": "native_unpatched_llama_forward",
            }
            records = [{
                **common,
                "condition": BASELINE,
                "layer": None,
                "ranking": None,
                "seed": 0,
                "intervention_token_logprob": baseline_logprob,
                "intervention_token_margin": baseline_margin,
                "argmax_token_id": baseline_token,
                "argmax_flipped": False,
                "js_divergence": 0.0,
                "final_prompt_hidden_cosine": 1.0,
                "latency_s": baseline["latency_s"],
                "actual_deleted_spatial_tokens": 0,
                "selected_original_indices": baseline_selected,
                "prefill_layer_input_lengths": baseline["layer_lengths"],
                "logit_distribution_sha256": (
                    baseline_distribution_hash
                ),
                "forward_path": "native_unpatched_llama_forward",
            }]

            for condition in condition_names()[1:]:
                arm = arms[condition]
                configure_arm(
                    model,
                    arm,
                    spatial_start,
                    spatial_tokens,
                )
                seed_all(question_arm_seed(question_id, arm["seed"]))
                intervention = forward_last(
                    model,
                    prepared,
                    trace,
                    geogate_forward,
                )
                selected_indices = model.model._gg_orig_map.detach().cpu()
                selected_original_indices = selected_indices.tolist()
                if (
                    not selected_original_indices
                    or selected_original_indices[-1] != full_tokens - 1
                ):
                    raise RuntimeError("Final prompt token was not preserved")
                intervention_token = int(
                    torch.argmax(intervention["logits"]).item()
                )
                intervention_logprob = float(
                    intervention["log_probs"][baseline_token].item()
                )
                intervention_margin = target_margin(
                    intervention["logits"],
                    baseline_token,
                )
                cosine = torch.nn.functional.cosine_similarity(
                    baseline["hidden"],
                    intervention["hidden"],
                    dim=0,
                    eps=1e-8,
                )
                values = (
                    intervention_logprob,
                    intervention_margin,
                    js_divergence(
                        baseline["log_probs"],
                        intervention["log_probs"],
                    ),
                    float(cosine.item()),
                    intervention["latency_s"],
                )
                if not all(math.isfinite(value) for value in values):
                    raise RuntimeError("Non-finite intervention endpoint")
                records.append({
                    **common,
                    "condition": condition,
                    "layer": arm["layer"],
                    "ranking": arm["ranking"],
                    "seed": arm["seed"],
                    "intervention_token_logprob": intervention_logprob,
                    "intervention_token_margin": intervention_margin,
                    "argmax_token_id": intervention_token,
                    "argmax_flipped": (
                        intervention_token != baseline_token
                    ),
                    "js_divergence": values[2],
                    "final_prompt_hidden_cosine": values[3],
                    "latency_s": values[4],
                    "actual_deleted_spatial_tokens": (
                        delete_count(spatial_tokens)
                    ),
                    "selected_original_indices": (
                        selected_original_indices
                    ),
                    "prefill_layer_input_lengths": (
                        intervention["layer_lengths"]
                    ),
                    "logit_distribution_sha256": tensor_sha256(
                        intervention["log_probs"]
                    ),
                    "forward_path": "geogate_intervention_forward",
                })
                del intervention

            if args.repeat_verification:
                verify_repeat(
                    model,
                    prepared,
                    trace,
                    native_forward,
                    geogate_forward,
                    question_id,
                    records,
                )
            if tensor_sha256(prepared[4]) != expanded_hash:
                raise RuntimeError(
                    "Prepared embedding changed within question group"
                )
            for record in records:
                record["repeat_verified"] = bool(
                    args.repeat_verification
                )
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            completed += 1
            del baseline, prepared, records, tensors, videos
            torch.cuda.empty_cache()
            print(
                f"{completed}/{len(selected_question_rows)} complete: "
                f"{question_id} ({scene_id})",
                flush=True,
            )

    status = {
        "state": "finished",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "is_full": is_full,
        "questions": completed,
        "scenes": selected_scene_count,
        "conditions": len(condition_names()),
        "rows": completed * len(condition_names()),
        "repeat_verification": bool(args.repeat_verification),
        "repeat_verified_questions": (
            completed if args.repeat_verification else 0
        ),
        "raw_sha256": digest_file(args.out),
        "manifest_sha256": digest_file(args.manifest),
    }
    write_json_exclusive(args.status, status)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
