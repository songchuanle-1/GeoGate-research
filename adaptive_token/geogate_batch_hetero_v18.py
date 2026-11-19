"""Heterogeneous mixed-scene GeoGate batching for variable visual counts.

This module is intentionally separate from ``geogate_batch_v6`` so previously
accepted v6/v7/v12 artifacts keep their frozen source hashes.
"""

import types

import torch
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
)
from transformers.modeling_outputs import BaseModelOutputWithPast

from adaptive_token.geogate_batch_v6 import (
    gather_cache,
    gather_sequence,
    infer_expanded_spans,
)


def build_heterogeneous_keep_indices(
    scores,
    spans,
    sequence_length,
    keep_ratio,
):
    """Return padded per-sample keep indices for variable visual counts."""
    if scores.ndim != 2 or scores.shape[0] != len(spans):
        raise ValueError("Scores must have shape (batch, sequence)")
    keep_counts = []
    rows = []
    for batch_index, (sp0, sp1) in enumerate(spans):
        visual_count = sp1 - sp0
        if visual_count <= 0:
            raise ValueError("Every sample requires at least one visual token")
        keep_count = max(1, int(round(float(keep_ratio) * visual_count)))
        keep_count = min(keep_count, visual_count)
        spatial_scores = scores[batch_index, sp0:sp1]
        keep_spatial = torch.sort(
            torch.topk(spatial_scores, keep_count).indices
        ).values + sp0
        rows.append(torch.cat([
            torch.arange(0, sp0, device=scores.device),
            keep_spatial,
            torch.arange(sp1, sequence_length, device=scores.device),
        ]))
        keep_counts.append(keep_count)
    valid_lengths = [int(row.numel()) for row in rows]
    width = max(valid_lengths)
    indices = torch.zeros(
        (len(rows), width),
        dtype=torch.long,
        device=scores.device,
    )
    for batch_index, row in enumerate(rows):
        indices[batch_index, width - row.numel():] = row
    return indices, keep_counts, valid_lengths


def enable_auto_spans(model, image_token_index):
    """Infer one expanded spatial span per sample during multimodal prefill."""
    if getattr(model, "_gg_hetero_auto_spans_enabled", False):
        return model
    original_prepare = model.prepare_inputs_labels_for_multimodal
    language_model = model.model

    def prepare_with_spans(self, *args, **kwargs):
        result = original_prepare(*args, **kwargs)
        input_ids = args[0] if args else kwargs.get("input_ids")
        input_attention_mask = (
            args[2] if len(args) > 2 else kwargs.get("attention_mask")
        )
        expanded_attention_mask = result[2]
        inputs_embeds = result[4]
        if (
            torch.is_tensor(input_ids)
            and torch.is_tensor(inputs_embeds)
            and input_ids.ndim == 2
            and input_ids.shape[1] > 1
        ):
            spans, visual_counts = infer_expanded_spans(
                input_ids,
                input_attention_mask,
                expanded_attention_mask,
                inputs_embeds.shape[1],
                image_token_index,
            )
            language_model._gg_hetero_spans = spans
            language_model._gg_hetero_visual_counts = visual_counts
        return result

    model.prepare_inputs_labels_for_multimodal = types.MethodType(
        prepare_with_spans,
        model,
    )
    model._gg_hetero_auto_spans_enabled = True
    return model


def enable_geogate_batch_hetero(model, prune_layer=14, keep_ratio=0.10):
    """Patch fixed single-drop GeoGate for mixed-scene tensor batches."""
    language_model = model.model
    language_model._gg_hetero_prune_layer = int(prune_layer)
    language_model._gg_hetero_keep_ratio = float(keep_ratio)
    language_model._gg_hetero_spans = None
    language_model._gg_hetero_visual_counts = None
    language_model._gg_hetero_next_pos = None
    language_model._gg_hetero_cache_mask = None
    language_model._gg_hetero_trace = []

    for layer in language_model.layers:
        rotary = layer.self_attn.rotary_emb
        if getattr(rotary, "_gg_hetero_wrapped", False):
            continue
        original_rotary = rotary.forward

        def wrapped(
            x,
            seq_len=None,
            _original=original_rotary,
            _lm=language_model,
        ):
            need = max(seq_len or 0, getattr(_lm, "_gg_hetero_maxpos", 0))
            return _original(x, seq_len=need)

        rotary.forward = wrapped
        rotary._gg_hetero_wrapped = True

    language_model.forward = types.MethodType(
        _heterogeneous_forward,
        language_model,
    )

    if not getattr(model, "_gg_hetero_generation_wrapped", False):
        original_prepare = model.prepare_inputs_for_generation

        def prepare_generation(
            input_ids,
            past_key_values=None,
            inputs_embeds=None,
            **kwargs,
        ):
            output = original_prepare(
                input_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
            if (
                past_key_values is not None
                and language_model._gg_hetero_next_pos is not None
            ):
                prepared_ids = output.get("input_ids")
                if prepared_ids is not None and prepared_ids.shape[1] > 1:
                    output["input_ids"] = prepared_ids[:, -1:]
                if output.get("position_ids") is not None:
                    output["position_ids"] = None
            return output

        model.prepare_inputs_for_generation = prepare_generation
        model._gg_hetero_generation_wrapped = True
    return model


def _heterogeneous_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    """Transformers 4.37 LlamaModel forward with variable kept widths."""
    from transformers.cache_utils import Cache, DynamicCache

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (
        return_dict
        if return_dict is not None
        else self.config.use_return_dict
    )

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("Specify either input_ids or inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    batch, sequence_length, _ = inputs_embeds.shape

    past_length = 0
    use_legacy_cache = False
    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_length = past_key_values.get_usable_length(sequence_length)

    is_prefill = past_length == 0
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch, sequence_length),
            dtype=torch.long,
            device=inputs_embeds.device,
        )
    if position_ids is None:
        position_ids = torch.arange(
            past_length,
            sequence_length + past_length,
            dtype=torch.long,
            device=inputs_embeds.device,
        ).unsqueeze(0).expand(batch, -1)

    spans = self._gg_hetero_spans
    prune_prefill = is_prefill and spans is not None
    if is_prefill:
        self._gg_hetero_trace = []
        self._gg_hetero_maxpos = int(position_ids.max().item()) + 641
        self._gg_hetero_next_pos = position_ids.max(dim=1).values + 1
        self._gg_hetero_cache_mask = None

    if not is_prefill and use_cache:
        real_past = past_key_values.get_seq_length()
        past_length = real_past
        if self._gg_hetero_cache_mask is None:
            raise RuntimeError("Missing pruned cache mask during batch decode")
        current_mask = torch.ones(
            (batch, sequence_length),
            dtype=self._gg_hetero_cache_mask.dtype,
            device=inputs_embeds.device,
        )
        attention_mask = torch.cat(
            [self._gg_hetero_cache_mask, current_mask],
            dim=1,
        )
        position_ids = self._gg_hetero_next_pos[:, None] + torch.arange(
            sequence_length,
            device=inputs_embeds.device,
        )[None, :]
        self._gg_hetero_next_pos = (
            self._gg_hetero_next_pos + sequence_length
        )

    attention_mask_4d = _prepare_4d_causal_attention_mask(
        attention_mask,
        (batch, sequence_length),
        inputs_embeds,
        past_length,
    )
    hidden_states = inputs_embeds
    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None
    keep_indices = None
    pre_prune_length = sequence_length

    for layer_index, layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        need_attention = (
            output_attentions
            or (
                prune_prefill
                and layer_index == self._gg_hetero_prune_layer
            )
        )
        layer_output = layer(
            hidden_states,
            attention_mask=attention_mask_4d,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=need_attention,
            use_cache=use_cache,
        )
        hidden_states = layer_output[0]
        if output_attentions:
            all_attentions += (layer_output[1],)

        if (
            prune_prefill
            and layer_index == self._gg_hetero_prune_layer
        ):
            attention = layer_output[1].float().mean(dim=1)
            query_indices = attention_mask.sum(dim=1) - 1
            query_indices = (
                query_indices
                + (sequence_length - attention_mask.sum(dim=1))
            )
            scores = attention[
                torch.arange(batch, device=attention.device),
                query_indices,
                :,
            ]
            (
                keep_indices,
                keep_counts,
                valid_lengths,
            ) = build_heterogeneous_keep_indices(
                scores,
                spans,
                sequence_length,
                self._gg_hetero_keep_ratio,
            )
            hidden_states = gather_sequence(hidden_states, keep_indices)
            position_ids = gather_sequence(position_ids, keep_indices)
            attention_mask = gather_sequence(attention_mask, keep_indices)
            padded_width = hidden_states.shape[1]
            for batch_index, valid_length in enumerate(valid_lengths):
                padding_before = padded_width - valid_length
                if padding_before:
                    attention_mask[batch_index, :padding_before] = 0
            sequence_length = padded_width
            attention_mask_4d = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch, sequence_length),
                hidden_states,
                0,
            )
            self._gg_hetero_trace.append({
                "event": "prune",
                "layer": layer_index,
                "batch": batch,
                "before": pre_prune_length,
                "after": sequence_length,
                "visual_counts": list(self._gg_hetero_visual_counts),
                "keep_counts": list(keep_counts),
                "valid_after": list(valid_lengths),
                "padding_before": [
                    sequence_length - value for value in valid_lengths
                ],
            })

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if prune_prefill and keep_indices is not None and use_cache:
        for layer_index in range(len(past_key_values.key_cache)):
            key = past_key_values.key_cache[layer_index]
            value = past_key_values.value_cache[layer_index]
            cache_length = key.shape[2]
            if cache_length == sequence_length:
                continue
            if cache_length != pre_prune_length:
                raise RuntimeError(
                    f"Unexpected KV length {cache_length} at layer "
                    f"{layer_index}; expected {pre_prune_length} or "
                    f"{sequence_length}"
                )
            past_key_values.key_cache[layer_index] = gather_cache(
                key,
                keep_indices,
            )
            past_key_values.value_cache[layer_index] = gather_cache(
                value,
                keep_indices,
            )
        self._gg_hetero_cache_mask = attention_mask
        self._gg_hetero_spans = None
    elif not is_prefill and use_cache:
        self._gg_hetero_cache_mask = attention_mask

    next_cache = None
    if use_cache:
        next_cache = (
            past_key_values.to_legacy_cache()
            if use_legacy_cache
            else past_key_values
        )
    if not return_dict:
        return tuple(
            value
            for value in [
                hidden_states,
                next_cache,
                all_hidden_states,
                all_attentions,
            ]
            if value is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_attentions,
    )
