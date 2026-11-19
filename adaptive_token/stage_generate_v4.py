"""3D-LLaVA generation callback for an already prepared, shared question."""

import hashlib
import random

import numpy as np

from adaptive_token.stage_arm_v4 import configure_arm


def tensor_hash(tensor):
    import torch
    return hashlib.sha256(
        tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def make_generate_arm(model, tokenizer, cache, trace, input_ids, generate_kwargs,
                      question_id, scene_id, prompt, spatial_start, input_hashes):
    import torch
    if cache.prepared is None:
        raise ValueError("Prepare the shared question first")
    embedding = cache.prepared[4]
    full = int(embedding.shape[1])
    spatial = full - int(input_ids.shape[1]) + 1
    if not (spatial > 0 and 0 <= spatial_start and spatial_start + spatial <= full):
        raise ValueError("Invalid expanded spatial span")
    original_hash = tensor_hash(embedding)

    def generate(arm):
        if tensor_hash(embedding) != original_hash:
            raise ValueError("Shared embedding changed before generation")
        random.seed(arm["seed"])
        np.random.seed(arm["seed"])
        torch.manual_seed(arm["seed"])
        torch.cuda.manual_seed_all(arm["seed"])
        counts = configure_arm(model, arm, spatial_start, spatial)
        trace.reset()
        hits = cache.hits
        with torch.inference_mode():
            output = model.generate(
                input_ids, do_sample=False, temperature=1.0, num_beams=1,
                max_new_tokens=64, tokenizer=tokenizer, use_cache=True,
                **generate_kwargs)
        if cache.hits != hits + 1:
            raise ValueError("Generation did not reuse exactly one cached prefill")
        if tensor_hash(embedding) != original_hash:
            raise ValueError("Shared embedding mutated by generation")
        selected = model.model._gg_orig_map.detach().cpu().tolist()
        actual_keep = sum(spatial_start <= i < spatial_start + spatial for i in selected)
        if actual_keep != counts["keep"]:
            raise ValueError("Observed spatial budget differs from configured budget")
        return {
            "question_id": str(question_id), "scene_id": scene_id,
            "condition": arm["name"], "seed": arm["seed"], "prompt": prompt,
            "text": tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip(),
            "input_tensor_hashes": dict(input_hashes),
            "expanded_input_sha256": original_hash, "full_tokens": full,
            "spatial_start": spatial_start, "input_visual_tokens": spatial,
            "actual_kept_spatial_tokens": actual_keep,
            "selected_original_indices": selected,
            "layer_trace": list(trace.layers),
            "kv_trace": [list(lengths) for lengths in trace.caches],
        }

    return generate
