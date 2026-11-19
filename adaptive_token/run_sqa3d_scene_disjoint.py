"""Resumable paired runner for the frozen SQA3D scene-disjoint experiment."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import time

import numpy as np
import torch

from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_special_token
from llava.model.builder import load_pretrained_model
from llava.pc_utils import Compose, vqa_transform_eval
from llava.utils import disable_torch_init
from pointgroup_ops import voxelization_idx

from adaptive_token.geogate_v0 import enable_geogate, set_span
from adaptive_token.gpu_guard import guard_gpu


TEMPLATE = (
    "<image>\n {question} Answer the question using a single word or phrase."
)


def load_questions(path: pathlib.Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8")
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError:
        rows = [
            json.loads(line)
            for line in raw.splitlines()
            if line.strip()
        ]
    if not isinstance(rows, list):
        raise ValueError("Question file must contain a JSON list or JSONL")
    return rows


def completed_ids(path: pathlib.Path) -> set[str]:
    if not path.is_file():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        question_id = str(row["question_id"])
        if question_id in completed:
            raise ValueError(f"Duplicate completed question: {question_id}")
        completed.add(question_id)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", required=True)
    parser.add_argument("--scan-folder", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--answers-file", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--model-variant", default="v0_base")
    parser.add_argument("--prune-layer", type=int, default=-1)
    parser.add_argument("--keep-ratio", type=float, default=0.10)
    parser.add_argument("--scorer", default="attn")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--guard-every", type=int, default=1)
    parser.add_argument("--guard-gpu-id", type=int, default=-1)
    args = parser.parse_args()
    if args.guard_every < 1:
        raise ValueError("--guard-every must be positive")
    if args.guard_gpu_id < 0:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not visible:
            raise ValueError(
                "--guard-gpu-id is required when CUDA_VISIBLE_DEVICES is unset"
            )
        devices = [item.strip() for item in visible.split(",") if item.strip()]
        if len(devices) != 1:
            raise ValueError("The guarded runner requires one visible GPU")
        args.guard_gpu_id = int(devices[0])

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    questions = load_questions(pathlib.Path(args.question_file))
    output_path = pathlib.Path(args.answers_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = completed_ids(output_path)
    question_ids = {str(row["question_id"]) for row in questions}
    unknown = completed - question_ids
    if unknown:
        raise ValueError(f"Completed IDs outside question set: {len(unknown)}")
    pending = [
        row for row in questions
        if str(row["question_id"]) not in completed
    ]
    print(
        f"questions={len(questions)} completed={len(completed)} "
        f"pending={len(pending)}"
    )
    if not pending:
        return

    guard_gpu(
        args.guard_gpu_id,
        {os.getpid()},
        "SQA3D scene-disjoint before model load",
    )
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, _, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        pointcloud_tower_name=None,
        model_variant=args.model_variant,
    )
    model.eval()
    condition = "baseline" if args.prune_layer < 0 else "geogate"
    if args.prune_layer >= 0:
        enable_geogate(
            model,
            prune_layer=args.prune_layer,
            keep_ratio=args.keep_ratio,
            scorer=args.scorer,
        )
    print(
        f"condition={condition} prune_layer={args.prune_layer} "
        f"keep_ratio={args.keep_ratio} scorer={args.scorer}"
    )

    scan_folder = pathlib.Path(args.scan_folder)
    with output_path.open("a", encoding="utf-8") as handle:
        for index, source in enumerate(pending, start=1):
            if index == 1 or index % args.guard_every == 0:
                guard_gpu(
                    args.guard_gpu_id,
                    {os.getpid()},
                    f"SQA3D scene-disjoint question {index}",
                )
            question_id = str(source["question_id"])
            scene = str(source["scene_id"])
            raw = torch.load(
                scan_folder / f"{scene}.pth",
                weights_only=False,
            )
            sp_mask = np.fromfile(
                scan_folder / "../super_points" / f"{scene}.bin",
                dtype=np.int64,
            )
            prepared = Compose(vqa_transform_eval)(
                dict(
                    scene_id=scene,
                    coord=raw["coord"],
                    color=raw["color"],
                    superpoint_mask=sp_mask,
                )
            )
            grid_coord = torch.cat(
                [
                    torch.LongTensor(
                        prepared["grid_coord"].shape[0], 1
                    ).fill_(0),
                    prepared["grid_coord"],
                ],
                1,
            )
            spatial_shape = np.clip(
                (grid_coord.max(0)[0][1:] + 1).numpy(),
                128,
                None,
            )
            voxel_coords, p2v_map, v2p_map = voxelization_idx(
                grid_coord, 1, 4
            )

            prompt_text = TEMPLATE.format(question=source["text"])
            conversation = conv_templates[args.conv_mode].copy()
            conversation.append_message(conversation.roles[0], prompt_text)
            conversation.append_message(conversation.roles[1], None)
            prompt = conversation.get_prompt()
            device = model.device
            input_ids = tokenizer_special_token(
                prompt,
                tokenizer,
                return_tensors="pt",
            ).unsqueeze(0).to(device)
            if args.prune_layer >= 0:
                image_positions = (input_ids[0] < 0).nonzero()
                image_pos = (
                    int(image_positions[0, 0].item())
                    if image_positions.numel()
                    else 0
                )
                set_span(model, image_pos, image_pos + 100)

            kwargs = dict(
                coord=prepared["coord"].to(device, dtype=torch.bfloat16),
                grid_coord=voxel_coords.to(device),
                offset=(
                    prepared["offset"].to(device)
                    if isinstance(prepared.get("offset"), torch.Tensor)
                    else torch.tensor(
                        [prepared["coord"].shape[0]],
                        device=device,
                    )
                ),
                feat=prepared["feat"].to(device, dtype=torch.bfloat16),
                p2v_map=p2v_map.to(device),
                v2p_map=v2p_map.to(device),
                spatial_shape=spatial_shape,
                superpoint_mask=[
                    torch.tensor(sp_mask).to(device)
                ],
                conditions=[prepared["condition"]],
                click_mask=[[]],
            )
            started = time.time()
            with torch.inference_mode():
                output = model.generate(
                    input_ids,
                    do_sample=False,
                    temperature=1.0,
                    num_beams=1,
                    max_new_tokens=args.max_new_tokens,
                    tokenizer=tokenizer,
                    use_cache=True,
                    **kwargs,
                )
            latency = time.time() - started
            text = tokenizer.batch_decode(
                output,
                skip_special_tokens=True,
            )[0].strip()
            row = {
                "question_id": question_id,
                "scene_id": scene,
                "prompt": source["text"],
                "text": text,
                "latency_s": round(latency, 3),
                "condition": condition,
                "prune_layer": int(args.prune_layer),
                "keep_ratio": float(args.keep_ratio),
                "scorer": str(args.scorer),
                "seed": int(args.seed),
                "model_id": (
                    "3D-LLaVA-7B-LoRA"
                    if args.prune_layer < 0
                    else "3D-LLaVA-7B-LoRA+geogate"
                ),
            }
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            if index % 50 == 0 or index == len(pending):
                print(
                    f"completed_this_run={index}/{len(pending)} "
                    f"question_id={question_id}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
