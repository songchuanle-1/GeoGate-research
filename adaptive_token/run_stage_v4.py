"""Assemble the frozen 3D-LLaVA stage experiment; launch only via idle guard."""

import argparse
import inspect
import importlib.metadata
import json
from pathlib import Path
import random

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", required=True)
    parser.add_argument("--scan-folder", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("delete10", "retain10"), default="delete10")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    import torch
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from llava.utils import disable_torch_init
    from adaptive_token.geogate_v0 import enable_geogate
    from adaptive_token.stage_pointcloud_v4 import load_question, preparation_kwargs
    from adaptive_token.stage_prepared_cache_v4 import StagePreparedCache
    from adaptive_token.stage_generate_v4 import make_generate_arm
    from adaptive_token.stage_question_v4 import execute_question
    from analysis.stage_protocol_v4 import arms
    from analysis.stage_trace_v4 import StageForwardTrace
    from analysis.stage_provenance_v4 import (
        dependency_manifest, pointcloud_dependencies, imported_dependency_files)
    import llava
    import pointgroup_ops
    from llava.pc_utils import Compose, vqa_transform_eval
    from analysis.review_v3_protocol import sample_seed
    from analysis.review_v4_journal import QuestionJournal

    questions = [json.loads(line) for line in args.questions.read_text().splitlines() if line.strip()]
    ids = [str(q["question_id"]) for q in questions]
    if len(ids) != 4675 or len(set(ids)) != 4675:
        raise ValueError("Expected full 4675 unique question set")
    if args.smoke:
        scenes = ("scene0011_00", "scene0231_00", "scene0249_00", "scene0342_00")
        questions = [next(q for q in questions if q["scene_id"] == s) for s in scenes]
    random.seed(20260908)
    np.random.seed(20260908)
    torch.manual_seed(20260908)
    torch.cuda.manual_seed_all(20260908)
    disable_torch_init()
    tokenizer, model, _, _ = load_pretrained_model(
        args.model_path, args.model_base, get_model_name_from_path(args.model_path),
        pointcloud_tower_name=None, model_variant="v0_base")
    model.eval()
    tower = model.get_pointcloud_tower()
    encoder = inspect.getfile(type(tower))
    weight, config = pointcloud_dependencies(tower.pointcloud_tower_name, encoder)
    root = Path(__file__).resolve().parents[1]
    sources = list((root / "adaptive_token").glob("stage_*.py"))
    sources += list((root / "analysis").glob("stage_*.py"))
    sources += [Path(__file__), root / "adaptive_token/geogate_v0.py",
                root / "adaptive_token/coverage_selector.py",
                root / "analysis/review_v4_journal.py", root / "analysis/review_v3_protocol.py",
                Path(encoder), config, Path(inspect.getfile(load_pretrained_model)),
                Path(inspect.getfile(type(model))),
                Path(inspect.getfile(model.prepare_inputs_labels_for_multimodal))]
    stack_root = Path(llava.__file__).resolve().parents[1]
    sources += imported_dependency_files(
        [root, stack_root, Path(pointgroup_ops.__file__).resolve().parent])
    sources = sorted(set(sources))
    data_files = [args.questions]
    for scene in sorted({q["scene_id"] for q in questions}):
        data_files += [args.scan_folder / f"{scene}.pth",
                       args.scan_folder / "../super_points" / f"{scene}.bin"]
    manifest = dependency_manifest(
        {"base": args.model_base, "adapter": args.model_path, "pointcloud": weight},
        sources, data_files)
    manifest.update(mode=args.mode, smoke=args.smoke, arms=arms(args.mode),
                    preparation_seed=20260908, max_new_tokens=64,
                    config=model.config.to_dict(),
                    packages={p: importlib.metadata.version(p)
                              for p in ("torch", "transformers", "numpy")})
    journal = QuestionJournal(args.out.with_suffix(".journal"), manifest,
                              [str(q["question_id"]) for q in questions],
                              [a["name"] for a in arms(args.mode)])
    enable_geogate(model, prune_layer=14, keep_ratio=0.1)
    cache, trace = StagePreparedCache(model), StageForwardTrace(model)
    completed = journal.completed()
    try:
        for index, question in enumerate(questions):
            qid = str(question["question_id"])
            if qid in completed:
                continue
            seed = sample_seed(qid, 20260908)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            ids, prompt, start, kwargs, hashes = load_question(
                question, args.scan_folder, model, tokenizer)
            with torch.inference_mode():
                cache.prepare(ids, **preparation_kwargs(kwargs))
            callback = make_generate_arm(
                model, tokenizer, cache, trace, ids, kwargs, qid,
                question["scene_id"], prompt, start, hashes)
            execute_question(qid, question["scene_id"], args.mode, index, callback, journal)
            cache.clear()
            del callback, kwargs, ids
            print(f"{index + 1}/{len(questions)} committed {qid}", flush=True)
        journal.export(args.out)
    finally:
        trace.close()
        cache.close()


if __name__ == "__main__":
    main()
