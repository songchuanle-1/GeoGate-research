#!/usr/bin/env python3
"""E1-b: layer-wise visual(superpoint)-token attention profile for 3D-LLaVA.

Same statistics as analysis/e1_attention_profile.py (LLaVA-3D), computed on the
superpoint-token architecture for cross-architecture comparison:
  - frac of gen-position attention mass on visual tokens, per layer
  - top50/top90 coverage among visual tokens
  - gini of received attention over visual tokens

Run from third_party/3D-LLaVA with its PYTHONPATH conventions (see smoke cmd in
results/BASELINE_SCANQA.md notes).
"""
import argparse
import json
import pathlib

import numpy as np
import torch

from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_special_token
from llava.model.builder import load_pretrained_model
from llava.pc_utils import Compose, vqa_transform_eval
from llava.utils import disable_torch_init
from pointgroup_ops import voxelization_idx

TEMPLATE = "<image>\n {question} Answer the question simply."


def ponder_collate(x):
    return torch.cat([x]) if isinstance(x, torch.Tensor) else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', required=True)
    ap.add_argument('--model-base', required=True)
    ap.add_argument('--scan-folder', required=True)
    ap.add_argument('--question-file', required=True)
    ap.add_argument('--num-samples', type=int, default=50)
    ap.add_argument('--dump-per-sample', action='store_true')
    ap.add_argument('--conv-mode', default='vicuna_v1')
    ap.add_argument('--model-variant', default='v0_base')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, _, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name,
        pointcloud_tower_name=None, model_variant=args.model_variant)
    model.eval()

    questions = [json.loads(q) for q in open(args.question_file)]
    seen, samples = set(), []
    for q in questions:
        if q['scene_id'] in seen:
            continue
        seen.add(q['scene_id'])
        samples.append(q)
        if len(samples) >= args.num_samples:
            break

    n_layers = model.config.num_hidden_layers
    agg = {k: [[] for _ in range(n_layers)]
           for k in ('frac', 'top50', 'top90', 'gini')}
    n_vis_list = []
    per_sample = []

    for si, source in enumerate(samples):
        scan_file = source['scene_id']
        scan_folder = pathlib.Path(args.scan_folder)
        raw = torch.load(scan_folder / f'{scan_file}.pth', weights_only=False)
        sp_mask = np.fromfile(scan_folder / '../super_points' / f'{scan_file}.bin',
                              dtype=np.int64)
        d = Compose(vqa_transform_eval)(dict(scene_id=scan_file, coord=raw['coord'],
                                             color=raw['color'], superpoint_mask=sp_mask))
        grid_coord = torch.cat(
            [torch.LongTensor(d['grid_coord'].shape[0], 1).fill_(0), d['grid_coord']], 1)
        spatial_shape = np.clip((grid_coord.max(0)[0][1:] + 1).numpy(), 128, None)
        voxel_coords, p2v_map, v2p_map = voxelization_idx(grid_coord, 1, 4)

        prompt_q = TEMPLATE.format(question=source['text'])
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt_q)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        dev = model.device
        input_ids = tokenizer_special_token(prompt, tokenizer,
                                            return_tensors='pt').unsqueeze(0).to(dev)

        with torch.inference_mode():
            prep = model.prepare_inputs_labels_for_multimodal(
                input_ids, None, None, None, labels=None,
                coord=d['coord'].to(dev, dtype=torch.bfloat16),
                grid_coord=voxel_coords.to(dev),
                offset=d['offset'].to(dev) if isinstance(d.get('offset'), torch.Tensor)
                       else torch.tensor([d['coord'].shape[0]], device=dev),
                pc_input=d['feat'].to(dev, dtype=torch.bfloat16),
                p2v_map=p2v_map.to(dev), v2p_map=v2p_map.to(dev),
                spatial_shape=spatial_shape,
                superpoint_mask=[torch.tensor(sp_mask).to(dev)],
                click_mask=[[]])
            # tuple layout follows llava_arch: (_, position_ids, attn_mask, _, inputs_embeds, labels, ...)
            position_ids, attention_mask, inputs_embeds = prep[1], prep[2], prep[4]
            total_len = inputs_embeds.shape[1]
            n_text = input_ids.shape[1] - 1          # minus the <image> placeholder
            n_vis = total_len - n_text
            img_pos = (input_ids[0] < 0).nonzero()[0, 0].item()  # special tokens are negative ids
            sp0, sp1 = img_pos, img_pos + n_vis
            out = model.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                              position_ids=position_ids, output_attentions=True,
                              use_cache=False, return_dict=True)

        n_vis_list.append(n_vis)
        rec = {
            'question_id': source['question_id'],
            'scene_id': scan_file,
            'text': source['text'],
            'n_visual': n_vis,
            'frac': [], 'top50': [], 'top90': [], 'gini': [],
        } if args.dump_per_sample else None
        for li, attn in enumerate(out.attentions):
            a = attn[0].float().mean(dim=0)
            gen_row = a[-1]
            to_sp = gen_row[sp0:sp1]
            frac = (to_sp.sum() / gen_row.sum()).item()
            agg['frac'][li].append(frac)
            if rec is not None:
                rec['frac'].append(round(frac, 6))
            if to_sp.sum() > 0:
                w = torch.sort(to_sp, descending=True).values
                c = torch.cumsum(w, 0) / w.sum()
                top50 = ((c < 0.5).sum().item() + 1) / n_vis
                top90 = ((c < 0.9).sum().item() + 1) / n_vis
                agg['top50'][li].append(top50)
                agg['top90'][li].append(top90)
                if rec is not None:
                    rec['top50'].append(round(top50, 6))
                    rec['top90'].append(round(top90, 6))
            recv = a[:, sp0:sp1].sum(dim=0)
            r, _ = torch.sort(recv)
            i = torch.arange(1, len(r) + 1, device=r.device, dtype=r.dtype)
            gini = (
                ((2 * i - len(r) - 1) * r).sum()
                / (len(r) * r.sum() + 1e-9)
            ).item()
            agg['gini'][li].append(gini)
            if rec is not None:
                rec['gini'].append(round(gini, 6))
        if rec is not None:
            per_sample.append(rec)
        del out
        torch.cuda.empty_cache()
        print(f'[{si + 1}/{len(samples)}] {scan_file} n_vis={n_vis}', flush=True)

    result = {
        'model': '3D-LLaVA-7B-LoRA', 'n_samples': len(samples), 'n_layers': n_layers,
        'n_visual_mean': float(np.mean(n_vis_list)),
        'layers': [{
            'layer': li,
            'frac_attn_to_visual': float(np.mean(agg['frac'][li])),
            'top50_coverage': float(np.mean(agg['top50'][li])) if agg['top50'][li] else None,
            'top90_coverage': float(np.mean(agg['top90'][li])) if agg['top90'][li] else None,
            'gini': float(np.mean(agg['gini'][li])),
        } for li in range(n_layers)],
    }
    if args.dump_per_sample:
        result['per_sample'] = per_sample
    json.dump(result, open(args.out, 'w'), indent=1)
    print('saved ->', args.out)


if __name__ == '__main__':
    main()
