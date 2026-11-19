"""Deterministic OpenEQA-HM3D RGB-D episode loading for v14."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


HM3D_DEPTH_UNITS_PER_METER = 65535.0 / 10.0


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def uniform_frame_indices(frame_count, target_count):
    """Return deterministic uniform indices, repeating only when necessary."""
    frame_count = int(frame_count)
    target_count = int(target_count)
    if frame_count < 1:
        raise ValueError("An episode must contain at least one RGB frame")
    if target_count < 1:
        raise ValueError("target_count must be positive")
    return [
        min(frame_count - 1, (index * frame_count) // target_count)
        for index in range(target_count)
    ]


def discover_hm3d_frames(episode_dir):
    episode_dir = Path(episode_dir)
    rgb_paths = sorted(episode_dir.glob("*-rgb.png"))
    if not rgb_paths:
        raise FileNotFoundError(
            f"No *-rgb.png frames found in {episode_dir}"
        )
    bundles = []
    for rgb_path in rgb_paths:
        prefix = rgb_path.name[:-len("-rgb.png")]
        depth_path = episode_dir / f"{prefix}-depth.png"
        pose_path = episode_dir / f"{prefix}.txt"
        missing = [
            str(path)
            for path in (depth_path, pose_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Incomplete RGB-D frame {rgb_path.name}: {missing}"
            )
        bundles.append({
            "rgb": rgb_path,
            "depth": depth_path,
            "pose": pose_path,
        })
    return bundles


def select_hm3d_frames(episode_dir, target_count):
    bundles = discover_hm3d_frames(episode_dir)
    indices = uniform_frame_indices(len(bundles), target_count)
    selected = [bundles[index] for index in indices]
    payload = {
        "episode": Path(episode_dir).name,
        "available_frames": len(bundles),
        "target_frames": int(target_count),
        "indices": indices,
        "files": [
            {
                key: value.name
                for key, value in bundle.items()
            }
            for bundle in selected
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["selection_sha256"] = hashlib.sha256(encoded).hexdigest()
    return selected, payload


def resolve_episode_dir(episode_root, episode):
    root = Path(episode_root)
    episode_path = Path(str(episode).replace("\\", "/"))
    candidates = [root / episode_path]
    if episode_path.parts and root.name == episode_path.parts[0]:
        candidates.insert(0, root.joinpath(*episode_path.parts[1:]))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def tensor_sha256(tensor):
    import torch

    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy()
    return hashlib.sha256(raw.tobytes()).hexdigest()


def load_hm3d_episode(episode_dir, video_processor, target_count=None):
    """Load official extracted HM3D RGB, depth, pose, and intrinsics.

    OpenEQA stores HM3D depth as uint16 values spanning 0--10 meters. The
    upstream LLaVA-3D processor has no explicit HM3D branch and assumes other
    depth scales, so this adapter performs the conversion directly.
    """
    import torch

    episode_dir = Path(episode_dir)
    if target_count is None:
        target_count = int(video_processor.num_frames)
    selected, trace = select_hm3d_frames(episode_dir, target_count)

    intrinsic_path = episode_dir / "intrinsic_color.txt"
    if not intrinsic_path.is_file():
        raise FileNotFoundError(f"Missing {intrinsic_path}")
    intrinsic = np.loadtxt(intrinsic_path)

    images = []
    depths = []
    poses = []
    processed_intrinsic = None
    for bundle in selected:
        image = Image.open(bundle["rgb"]).convert("RGB")
        image_size = image.size
        image_tensor = video_processor.image_processor.preprocess(
            images=image,
            do_rescale=True,
            do_normalize=True,
            return_tensors="pt",
        )["pixel_values"][0]

        depth_image = Image.open(bundle["depth"])
        depth_array, resize_shape = video_processor.preprocess_depth_image(
            depth_image,
            do_depth_scale=False,
        )
        depth_tensor = torch.as_tensor(
            np.ascontiguousarray(depth_array)
        ).float() / HM3D_DEPTH_UNITS_PER_METER
        pose_tensor = torch.from_numpy(
            np.loadtxt(bundle["pose"])
        ).float()

        if processed_intrinsic is None:
            processed_intrinsic = video_processor.preprocess_instrinsic(
                intrinsic.copy(),
                image_size,
                resize_shape,
            )
        images.append(image_tensor)
        depths.append(depth_tensor)
        poses.append(pose_tensor)

    intrinsic_tensor = torch.from_numpy(
        np.asarray(processed_intrinsic)
    ).float()
    if intrinsic_tensor.ndim == 2:
        intrinsic_tensor = intrinsic_tensor.unsqueeze(0).repeat(
            len(selected),
            1,
            1,
        )
    elif intrinsic_tensor.shape[0] != len(selected):
        raise ValueError(
            "HM3D intrinsic count does not match selected frame count"
        )

    tensors = {
        "images": torch.stack(images).unsqueeze(0),
        "depths": torch.stack(depths).unsqueeze(0),
        "poses": torch.stack(poses).unsqueeze(0),
        "intrinsics": intrinsic_tensor.unsqueeze(0),
    }
    trace["file_sha256"] = {
        path.name: digest_file(path)
        for bundle in selected
        for path in bundle.values()
    }
    trace["intrinsic_file"] = intrinsic_path.name
    trace["intrinsic_sha256"] = digest_file(intrinsic_path)
    trace["tensor_sha256"] = {
        key: tensor_sha256(value)
        for key, value in tensors.items()
    }
    trace["depth_conversion"] = (
        "uint16 / (65535 / 10 meters)"
    )
    trace["pose_transform"] = "identity_axis_alignment"
    return tensors, trace
