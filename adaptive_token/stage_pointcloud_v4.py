"""Prepare one 3D-LLaVA ScanQA input using the existing evaluation transforms."""

from pathlib import Path


def validate_eval_condition(conditions):
    if conditions != ["textgen"]:
        raise ValueError("Expected upstream ScanQA evaluation textgen condition")


def load_question(source, scan_folder, model, tokenizer):
    import numpy as np
    import torch
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_special_token
    from llava.pc_utils import Compose, vqa_transform_eval
    from pointgroup_ops import voxelization_idx
    from adaptive_token.stage_generate_v4 import tensor_hash

    scene = source["scene_id"]
    root = Path(scan_folder)
    raw = torch.load(root / f"{scene}.pth", weights_only=False)
    mask = np.fromfile(root / "../super_points" / f"{scene}.bin", dtype=np.int64)
    data = Compose(vqa_transform_eval)(
        dict(scene_id=scene, coord=raw["coord"], color=raw["color"],
             superpoint_mask=mask))
    grid = torch.cat([torch.zeros((data["grid_coord"].shape[0], 1),
                                 dtype=torch.long), data["grid_coord"]], 1)
    shape = np.clip((grid.max(0)[0][1:] + 1).numpy(), 128, None)
    voxels, p2v, v2p = voxelization_idx(grid, 1, 4)
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0],
                        f"<image>\n {source['text']} Answer the question simply.")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    device = model.device
    ids = tokenizer_special_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(device)
    positions = (ids[0] == IMAGE_TOKEN_INDEX).nonzero()
    if positions.numel() != 1 or int((ids < 0).sum()) != 1:
        raise ValueError("Expected exactly one spatial placeholder")
    kwargs = dict(
        coord=data["coord"].to(device, dtype=torch.bfloat16),
        grid_coord=voxels.to(device),
        offset=data["offset"].to(device) if isinstance(data.get("offset"), torch.Tensor)
        else torch.tensor([data["coord"].shape[0]], device=device),
        feat=data["feat"].to(device, dtype=torch.bfloat16),
        p2v_map=p2v.to(device), v2p_map=v2p.to(device),
        spatial_shape=shape, superpoint_mask=[torch.tensor(mask).to(device)],
        conditions=[data["condition"]], click_mask=[[]])
    validate_eval_condition(kwargs["conditions"])
    hashes = {key: tensor_hash(value) for key, value in kwargs.items()
              if isinstance(value, torch.Tensor)}
    hashes["superpoint_mask"] = tensor_hash(kwargs["superpoint_mask"][0])
    hashes["spatial_shape"] = tensor_hash(torch.as_tensor(shape))
    return ids, prompt, int(positions[0, 0]), kwargs, hashes


def preparation_kwargs(generate_kwargs):
    result = dict(generate_kwargs)
    result["pc_input"] = result.pop("feat")
    result.pop("conditions")
    return result
