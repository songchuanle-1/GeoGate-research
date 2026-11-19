"""Explicit dependency hashing for stage runs, including base and point tower."""

import hashlib
import sys
from pathlib import Path


def imported_dependency_files(roots):
    """Record imported Python/native modules beneath explicit execution roots."""
    roots = [Path(root).resolve(strict=True) for root in roots]
    paths = set()
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.is_file() and any(path.is_relative_to(root) for root in roots):
            paths.add(path)
    return sorted(paths)


def pointcloud_dependencies(tower_name, encoder_source):
    """Mirror inspected 3D-LLaVA loader naming; fail if either file is absent."""
    name = str(tower_name)
    config_name = name.split("/")[-1].split(".")[0]
    config = Path(encoder_source).resolve(strict=True).parent / "configs" / f"{config_name}.py"
    checkpoint = Path(name.replace("_fp32", "")).resolve(strict=True)
    return checkpoint, config.resolve(strict=True)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def dependency_manifest(checkpoint_roots, source_files, data_files):
    """Call with resolved actual loader paths, not guessed configuration names."""
    if not {"base", "adapter", "pointcloud"}.issubset(checkpoint_roots):
        raise ValueError("Base, adapter and pointcloud checkpoint roots required")
    output = {"checkpoints": {}, "sources": {}, "data": {}}
    for role, root in checkpoint_roots.items():
        root = Path(root).resolve(strict=True)
        paths = [root] if root.is_file() else sorted(
            p for p in root.rglob("*") if p.is_file() and p.suffix in
            (".bin", ".pth", ".pt", ".safetensors", ".json", ".model"))
        if not paths:
            raise ValueError(f"No dependency files for {role}")
        output["checkpoints"][role] = {
            str(p): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in paths}
    for name, paths in (("sources", source_files), ("data", data_files)):
        paths = list(paths)
        if not paths:
            raise ValueError(f"Empty {name} dependency set")
        output[name] = {str(Path(p).resolve(strict=True)): digest(p) for p in paths}
    return output
