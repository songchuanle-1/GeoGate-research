#!/usr/bin/env python3
"""CPU asset preflight for the frozen OpenEQA-HM3D v14 protocol."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from adaptive_token.openeqa_hm3d_v14 import (
    digest_file,
    discover_hm3d_frames,
    resolve_episode_dir,
    select_hm3d_frames,
)


FROZEN_DATASET_SHA256 = (
    "afbcc703c50c121451a912181a4ae91f4ad94866b6dc48bdd7e9f566688fc8c9"
)
FROZEN_QUESTIONS = 557
FROZEN_EPISODES = 63
FROZEN_FRAMES = 20


def audit_episode(episode_dir):
    bundles = discover_hm3d_frames(episode_dir)
    selected, trace = select_hm3d_frames(
        episode_dir,
        FROZEN_FRAMES,
    )
    intrinsic_path = Path(episode_dir) / "intrinsic_color.txt"
    if not intrinsic_path.is_file():
        raise FileNotFoundError(f"Missing {intrinsic_path}")
    intrinsic = np.loadtxt(intrinsic_path)
    if intrinsic.shape != (4, 4) or not np.isfinite(intrinsic).all():
        raise ValueError(
            f"Invalid intrinsic matrix in {intrinsic_path}"
        )

    pose_determinants = []
    for bundle in bundles:
        pose = np.loadtxt(bundle["pose"])
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(f"Invalid pose matrix: {bundle['pose']}")
        determinant = float(np.linalg.det(pose[:3, :3]))
        if not math.isfinite(determinant) or abs(determinant) < 1e-6:
            raise ValueError(
                f"Singular pose rotation: {bundle['pose']}"
            )
        pose_determinants.append(determinant)

    rgb_sizes = set()
    depth_sizes = set()
    depth_min = None
    depth_max = None
    for bundle in selected:
        with Image.open(bundle["rgb"]) as image:
            rgb_sizes.add(tuple(image.size))
        with Image.open(bundle["depth"]) as image:
            depth_sizes.add(tuple(image.size))
            depth = np.asarray(image)
        if depth.ndim != 2 or not np.issubdtype(
            depth.dtype,
            np.integer,
        ):
            raise ValueError(
                f"Expected integer single-channel depth: {bundle['depth']}"
            )
        current_min = int(depth.min())
        current_max = int(depth.max())
        depth_min = (
            current_min if depth_min is None
            else min(depth_min, current_min)
        )
        depth_max = (
            current_max if depth_max is None
            else max(depth_max, current_max)
        )
    if len(rgb_sizes) != 1 or len(depth_sizes) != 1:
        raise ValueError(
            f"Inconsistent selected frame sizes in {episode_dir}"
        )
    if rgb_sizes != depth_sizes:
        raise ValueError(
            f"RGB/depth resolution mismatch in {episode_dir}"
        )
    if depth_min is None or depth_max is None or not (
        0 <= depth_min <= depth_max <= 65535
    ):
        raise ValueError(
            f"Invalid HM3D uint16 depth range in {episode_dir}"
        )
    return {
        "episode_dir": str(Path(episode_dir).resolve()),
        "available_frames": len(bundles),
        "selected_frames": FROZEN_FRAMES,
        "selection_sha256": trace["selection_sha256"],
        "selected_indices": trace["indices"],
        "resolution": list(next(iter(rgb_sizes))),
        "selected_depth_uint16_range": [depth_min, depth_max],
        "pose_rotation_determinant_range": [
            min(pose_determinants),
            max(pose_determinants),
        ],
        "intrinsic_sha256": digest_file(intrinsic_path),
    }


def preflight(dataset_path, episode_root):
    if digest_file(dataset_path) != FROZEN_DATASET_SHA256:
        raise ValueError("OpenEQA dataset does not match frozen SHA256")
    dataset = json.loads(
        Path(dataset_path).read_text(encoding="utf-8")
    )
    selected = [
        row for row in dataset
        if str(row.get("episode_history", "")).startswith("hm3d-v0/")
    ]
    episodes = list(dict.fromkeys(
        str(row["episode_history"]) for row in selected
    ))
    if (
        len(selected) != FROZEN_QUESTIONS
        or len(episodes) != FROZEN_EPISODES
    ):
        raise ValueError(
            "Expected frozen 557 HM3D questions and 63 episodes"
        )

    episode_rows = []
    errors = []
    for episode in episodes:
        episode_dir = resolve_episode_dir(episode_root, episode)
        try:
            row = audit_episode(episode_dir)
            row["episode"] = episode
            episode_rows.append(row)
        except Exception as error:
            errors.append({
                "episode": episode,
                "episode_dir": str(episode_dir),
                "error": f"{type(error).__name__}: {error}",
            })
    return {
        "state": "passed" if not errors else "failed",
        "dataset": {
            "path": str(Path(dataset_path).resolve()),
            "sha256": digest_file(dataset_path),
            "questions": len(selected),
            "episodes": len(episodes),
            "category_counts": dict(sorted(Counter(
                str(row.get("category", "unknown"))
                for row in selected
            ).items())),
        },
        "episode_root": str(Path(episode_root).resolve()),
        "frozen_frames": FROZEN_FRAMES,
        "episodes_passed": len(episode_rows),
        "episodes_failed": len(errors),
        "episode_rows": episode_rows,
        "errors": errors,
        "gpu_execution_authorized": not errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    result = preflight(args.dataset, args.episode_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["state"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
