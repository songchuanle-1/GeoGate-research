#!/usr/bin/env python3
"""GeoGate-instrumented Multi3DRefer evaluation for 3D-LLaVA.
Based on the original llava/eval/model_multi3drefer.py.
Adds GeoGate pruning while preserving the correct grounding pipeline
(model.generate returns binary mask, not text).

Multi3DRefer specifics vs ScanRefer:
  - object_ids (list) instead of object_id (single)
  - Zero-target cases (empty object_ids -> gt_mask all False)
  - Template includes "There may be no corresponding object..."
"""
import argparse, json, math, os, pathlib, time
import numpy as np
import torch
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_special_token, get_model_name_from_path
from llava.pc_utils import referseg_transform_eval, Compose
from pointgroup_ops import voxelization_idx
from typing import Sequence, Mapping

from adaptive_token.geogate_v0 import enable_geogate, set_span

TEMPLATE = "<image>\n Please output the segmentation mask according to the following description. \n{description}\nThere may be no corresponding object, or there may be one or more objects."


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


def ponder_collate_fn(batch, max_point=-1):
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")
    if max_point > 0:
        accum_num_points = 0
        ret_batches = []
        for data in batch:
            if accum_num_points + data["coord"].shape[0] > max_point:
                continue
            accum_num_points += data["coord"].shape[0]
            ret_batches.append(data)
        return ponder_collate_fn(ret_batches)
    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [ponder_collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {key: ponder_collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in batch.keys():
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)
        return batch
    else:
        from torch.utils.data.dataloader import default_collate
        return default_collate(batch)


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, _, _ = load_pretrained_model(
        model_path, args.model_base, model_name,
        pointcloud_tower_name=args.pointcloud_tower_name,
        model_variant=args.model_variant)

    # Enable GeoGate if requested
    if args.prune_layer >= 0:
        enable_geogate(model, prune_layer=args.prune_layer,
                       keep_ratio=args.keep_ratio)
        print(f'GeoGate ON: layer={args.prune_layer} keep={args.keep_ratio}')

    questions = json.load(open(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    ans_file = open(args.answers_file, 'w')

    for idx, source in enumerate(tqdm(questions)):
        scan_file = source['scene_id']
        scan_data_path = pathlib.Path(args.scan_folder) / f'{scan_file}.pth'
        superpoint_path = pathlib.Path(args.scan_folder) / '../super_points' / f'{scan_file}.bin'

        raw_data = torch.load(scan_data_path, weights_only=False)
        coord = raw_data['coord']
        color = raw_data['color']
        superpoint_mask = np.fromfile(superpoint_path, dtype=np.int64)

        # Multi3DRefer: object_ids is a list (multi-target, can be empty for zero-target)
        instance = raw_data['instance_gt']
        object_ids = [int(obj) for obj in source['object_ids']]
        gt_mask = np.isin(instance, object_ids)

        transform = Compose(referseg_transform_eval)
        pc_data_dict = dict(
            coord=coord, color=color, superpoint_mask=superpoint_mask)
        pc_data_dict = transform(pc_data_dict)

        grid_coord = pc_data_dict['grid_coord']
        grid_coord = torch.cat([torch.LongTensor(grid_coord.shape[0], 1).fill_(0), grid_coord], 1)
        pc_data_dict['grid_coord'] = grid_coord

        spatial_shape = np.clip((grid_coord.max(0)[0][1:] + 1).numpy(), 128, None)
        voxel_coords, p2v_map, v2p_map = voxelization_idx(grid_coord, 1, 4)

        for key in pc_data_dict:
            if key in ['coord', 'grid_coord', 'feat', 'offset']:
                pc_data_dict[key] = ponder_collate_fn([pc_data_dict[key]])

        qs = TEMPLATE.format(description=source['description'])
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        device = model.device
        input_ids = tokenizer_special_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0).to(device)

        # Set GeoGate spatial span over superpoint tokens (always 100 for 3D-LLaVA)
        if args.prune_layer >= 0:
            img_pos = (input_ids[0] < 0).nonzero()
            img_pos = img_pos[0, 0].item() if img_pos.numel() > 0 else 0
            set_span(model, img_pos, img_pos + 100)

        t0 = time.time()
        with torch.inference_mode():
            pred_mask = model.generate(
                input_ids,
                coord=pc_data_dict['coord'].to(device, dtype=torch.bfloat16),
                grid_coord=voxel_coords.to(device),
                offset=pc_data_dict['offset'].to(device),
                feat=pc_data_dict['feat'].to(device, dtype=torch.bfloat16),
                p2v_map=p2v_map.to(device),
                v2p_map=v2p_map.to(device),
                spatial_shape=spatial_shape,
                superpoint_mask=[torch.tensor(superpoint_mask).to(device)],
                conditions=[pc_data_dict['condition']],
                do_sample=False,
                temperature=1.0,
                num_beams=1,
                max_new_tokens=64,
                tokenizer=tokenizer,
                click_mask=[[]],
                use_cache=True)
        dt = time.time() - t0

        pred_mask = pred_mask.cpu().numpy().astype(bool)[0]
        gt_mask_arr = gt_mask.astype(bool)

        # Multi3DRefer IoU: handle zero-target cases (gt empty)
        if np.sum(gt_mask_arr) > 0:
            I = np.sum(np.logical_and(pred_mask, gt_mask_arr))
            U = np.sum(np.logical_or(pred_mask, gt_mask_arr))
            iou = 0.0 if U == 0 else float(I) / float(U)
        else:
            # Zero-target: correct answer is empty mask
            if np.sum(pred_mask) > 0:
                iou = 0.0
            else:
                iou = 1.0

        tp25 = 1 if iou >= 0.25 else 0
        tp50 = 1 if iou >= 0.50 else 0

        ans_file.write(json.dumps({
            'scene_id': scan_file, 'question_id': idx,
            'prompt': qs, 'model_id': model_name,
            'iou': iou, 'tp50': tp50, 'tp25': tp25,
            'latency_s': round(dt, 3),
            'object_ids': object_ids,
        }) + '\n')
        ans_file.flush()
    ans_file.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', type=str, default='facebook/opt-350m')
    ap.add_argument('--model-base', type=str, default=None)
    ap.add_argument('--pointcloud-tower-name', type=str, default=None)
    ap.add_argument('--scan-folder', type=str, default='')
    ap.add_argument('--question-file', type=str, default='tables/question.jsonl')
    ap.add_argument('--answers-file', type=str, default='answer.jsonl')
    ap.add_argument('--conv-mode', type=str, default='vicuna_v1')
    ap.add_argument('--num-chunks', type=int, default=1)
    ap.add_argument('--chunk-idx', type=int, default=0)
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--top_p', type=float, default=None)
    ap.add_argument('--num_beams', type=int, default=1)
    ap.add_argument('--model-variant', type=str, default='v0_base')
    ap.add_argument('--prune-layer', type=int, default=-1)
    ap.add_argument('--keep-ratio', type=float, default=0.10)
    ap.add_argument('--geo-beta', type=float, default=0.0)
    eval_model(ap.parse_args())
