#!/usr/bin/env python3
"""GeoGate on 3D-LLaVA (superpoint architecture). Readout-stage pruning at ~14th layer.

Same eval protocol as third_party/3D-LLaVA llava/eval/model_scanqa.py,
injecting enable_geogate + set_span.
"""
import argparse
import json
import pathlib
import random
import time

import numpy as np
import torch

from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_special_token
from llava.model.builder import load_pretrained_model
from llava.pc_utils import Compose, vqa_transform_eval
from llava.utils import disable_torch_init
from pointgroup_ops import voxelization_idx

from adaptive_token.adaptive_policy import question_keep_ratio
from adaptive_token.geogate_v0 import enable_geogate, set_keep_ratio, set_span

TEMPLATE = "<image>\n {question} Answer the question simply."


def ponder_collate_fn(batch, max_point=-1):
    if not isinstance(batch, (list, tuple)):
        raise TypeError(f"{batch} not supported")
    if isinstance(batch[0], torch.Tensor):
        return torch.cat(batch)
    if isinstance(batch[0], dict):
        return {k: ponder_collate_fn([d[k] for d in batch]) for k in batch[0]}
    from torch.utils.data.dataloader import default_collate
    return default_collate(batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', required=True)
    ap.add_argument('--model-base', required=True)
    ap.add_argument('--scan-folder', required=True)
    ap.add_argument('--question-file', required=True)
    ap.add_argument('--answers-file', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--conv-mode', default='vicuna_v1')
    ap.add_argument('--model-variant', default='v0_base')
    ap.add_argument('--prune-layer', type=int, default=-1)
    ap.add_argument('--keep-ratio', type=float, default=0.10)
    ap.add_argument('--max-keep-ratio', type=float, default=0.50)
    ap.add_argument('--dynamic-k', action='store_true',
                    help='Instruction-conditioned dynamic K')
    ap.add_argument('--adaptive-onset', action='store_true',
                    help='Detect the readout onset from attention concentration')
    ap.add_argument('--stability-window', type=int, default=3)
    ap.add_argument('--onset-min-layer', type=int, default=10)
    ap.add_argument('--onset-max-layer', type=int, default=18)
    ap.add_argument('--onset-patience', type=int, default=2)
    ap.add_argument('--onset-min-alpha', type=float, default=0.10)
    ap.add_argument('--onset-max-coverage', type=float, default=0.12)
    ap.add_argument('--geo-beta', type=float, default=0.0)
    ap.add_argument('--scorer', default='attn',
                    help='attn | random | prompt_mean | inverse_attn')
    ap.add_argument('--coverage-weight', type=float, default=0.25)
    ap.add_argument('--prune-mode', choices=('keep_topk', 'delete_topk'),
                    default='keep_topk')
    ap.add_argument('--delete-n', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, _, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name,
        pointcloud_tower_name=None, model_variant=args.model_variant)
    model.eval()
    pruning_enabled = args.prune_layer >= 0 or args.adaptive_onset
    if pruning_enabled:
        enable_geogate(
            model, prune_layer=args.prune_layer, keep_ratio=args.keep_ratio,
            scorer='stable_attn' if args.adaptive_onset else args.scorer,
            adaptive_onset=args.adaptive_onset,
            stability_window=args.stability_window,
            min_onset_layer=args.onset_min_layer,
            max_onset_layer=args.onset_max_layer,
            onset_patience=args.onset_patience,
            onset_min_alpha=args.onset_min_alpha,
            onset_max_coverage=args.onset_max_coverage,
            coverage_weight=args.coverage_weight,
            prune_mode=args.prune_mode,
            delete_n=args.delete_n,
        )
        if args.scorer in ('coverage_attn', 'stable_coverage'):
            from adaptive_token.geo_coords import attach_ost_coord_hook
            attach_ost_coord_hook(model)
        print(f'GeoGate ON: 3D-LLaVA layer={args.prune_layer} keep={args.keep_ratio} '
              f'scorer={args.scorer} adaptive_onset={args.adaptive_onset}')

    questions = [json.loads(q) for q in open(args.question_file)]
    ans_file = open(args.answers_file, 'w')

    for source in questions:
        sid = source['question_id']
        scene = source['scene_id']
        scan_folder = pathlib.Path(args.scan_folder)
        raw = torch.load(scan_folder / f'{scene}.pth', weights_only=False)
        sp_mask = np.fromfile(scan_folder / '../super_points' / f'{scene}.bin',
                              dtype=np.int64)
        d = Compose(vqa_transform_eval)(
            dict(scene_id=scene, coord=raw['coord'], color=raw['color'],
                 superpoint_mask=sp_mask))
        grid_coord = torch.cat(
            [torch.LongTensor(d['grid_coord'].shape[0], 1).fill_(0),
             d['grid_coord']], 1)
        spatial_shape = np.clip((grid_coord.max(0)[0][1:] + 1).numpy(), 128, None)
        voxel_coords, p2v_map, v2p_map = voxelization_idx(grid_coord, 1, 4)

        qs = TEMPLATE.format(question=source['text'])
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        dev = model.device
        input_ids = tokenizer_special_token(prompt, tokenizer,
                                            return_tensors='pt').unsqueeze(0).to(dev)

        used_ratio = args.keep_ratio
        if pruning_enabled:
            if args.dynamic_k:
                used_ratio = question_keep_ratio(
                    source['text'], args.keep_ratio, max_ratio=args.max_keep_ratio)
                set_keep_ratio(
                    model,
                    used_ratio,
                )
            else:
                set_keep_ratio(model, args.keep_ratio)
                used_ratio = args.keep_ratio
            # spatial span: <image> placeholder expands to 100 superpoint tokens
            img_pos = (input_ids[0] < 0).nonzero()
            if img_pos.numel() == 0:
                img_pos = 0
            else:
                img_pos = img_pos[0, 0].item()
            set_span(model, img_pos, img_pos + 100)

        kw = dict(
            coord=d['coord'].to(dev, dtype=torch.bfloat16),
            grid_coord=voxel_coords.to(dev),
            offset=d['offset'].to(dev) if isinstance(d.get('offset'), torch.Tensor)
                  else torch.tensor([d['coord'].shape[0]], device=dev),
            feat=d['feat'].to(dev, dtype=torch.bfloat16),
            p2v_map=p2v_map.to(dev), v2p_map=v2p_map.to(dev),
            spatial_shape=spatial_shape, superpoint_mask=[torch.tensor(sp_mask).to(dev)],
            conditions=[d['condition']],
            click_mask=[[]])
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(input_ids, do_sample=False, temperature=1.0,
                                 num_beams=1, max_new_tokens=64,
                                 tokenizer=tokenizer, use_cache=True, **kw)
        dt = time.time() - t0
        text = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
        ans_file.write(json.dumps({
            'question_id': sid, 'prompt': source['text'],
            'text': text, 'latency_s': round(dt, 3),
            'keep_ratio': round(float(used_ratio), 4),
            'prune_layer': int(args.prune_layer),
            'scorer': args.scorer,
            'stability_window': int(args.stability_window),
            'onset_layer': getattr(model.model, '_gg_onset_layer', None),
            'adaptive_onset': bool(args.adaptive_onset),
            'seed': int(args.seed),
            'prune_mode': args.prune_mode,
            'delete_n': int(args.delete_n),
            'model_id': '3D-LLaVA-7B-LoRA+adaptive_geogate'
            if args.adaptive_onset else
            ('3D-LLaVA-7B-LoRA+geogate' if pruning_enabled else '3D-LLaVA-7B-LoRA'),
        }) + '\n')
        ans_file.flush()
    ans_file.close()


if __name__ == '__main__':
    main()
