"""Trace and validate complete prefill/decode execution for v15."""

import hashlib
import json


class GenerationTrace:
    def __init__(self, model):
        self.layers = []
        self.caches = []
        self.handles = []
        decoder = model.model
        if len(decoder.layers) != 32:
            raise ValueError("Frozen v15 protocol requires 32 decoder layers")
        for index, block in enumerate(decoder.layers):
            def before(_module, args, kwargs, index=index):
                hidden = args[0] if args else kwargs["hidden_states"]
                self.layers.append((index, int(hidden.shape[1])))
            self.handles.append(
                block.register_forward_pre_hook(before, with_kwargs=True)
            )
        self.handles.append(decoder.register_forward_hook(self.after))

    def after(self, module, _args, output):
        cache = output.past_key_values
        if cache is None:
            raise ValueError("v15 generation requires a returned KV cache")
        if hasattr(cache, "get_seq_length"):
            lengths = [
                int(cache.get_seq_length(index))
                for index in range(len(module.layers))
            ]
        else:
            lengths = [int(entry[0].shape[-2]) for entry in cache]
        self.caches.append(lengths)

    def reset(self):
        self.layers.clear()
        self.caches.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def trace_sha256(layer_lengths, cache_lengths):
    payload = json.dumps(
        {"layers": layer_lengths, "caches": cache_lengths},
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def summarize_trace(trace, full_tokens, prune_layer=None, reduced_tokens=None):
    if not trace.caches or len(trace.layers) != 32 * len(trace.caches):
        raise ValueError("Incomplete v15 forward trace")
    chunks = [
        trace.layers[index:index + 32]
        for index in range(0, len(trace.layers), 32)
    ]
    if prune_layer is None:
        expected_prefill = [full_tokens] * 32
        first_cache = full_tokens
    else:
        if reduced_tokens is None:
            raise ValueError("Reduced length is required for an intervention")
        expected_prefill = (
            [full_tokens] * (prune_layer + 1)
            + [reduced_tokens] * (31 - prune_layer)
        )
        first_cache = reduced_tokens
    if chunks[0] != list(enumerate(expected_prefill)):
        raise ValueError("Unexpected v15 prefill layer lengths")
    for chunk in chunks[1:]:
        if chunk != list(enumerate([1] * 32)):
            raise ValueError("v15 decode is not single-token")
    cache_scalars = []
    for step, lengths in enumerate(trace.caches):
        expected = first_cache + step
        if lengths != [expected] * 32:
            raise ValueError("Unexpected v15 KV cache growth")
        cache_scalars.append(expected)
    layer_lengths = [
        [length for _, length in chunk]
        for chunk in chunks
    ]
    cache_lengths = [list(lengths) for lengths in trace.caches]
    return {
        "prefill_layer_input_lengths": layer_lengths[0],
        "decode_forward_calls": len(chunks) - 1,
        "decode_single_token_verified": True,
        "layer_input_lengths_by_forward": layer_lengths,
        "kv_cache_length_summary": cache_scalars,
        "kv_cache_lengths_by_forward": cache_lengths,
        "kv_all_32_layers_equal": True,
        "complete_trace_sha256": trace_sha256(
            layer_lengths,
            cache_lengths,
        ),
    }
