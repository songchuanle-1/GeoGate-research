"""Scene-bucketed fixed-schedule GeoGate for true tensor batching."""

import types

import torch
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
)
from transformers.modeling_outputs import BaseModelOutputWithPast


def infer_expanded_spans(
    input_ids,
    input_attention_mask,
    expanded_attention_mask,
    expanded_width,
    image_token_index,
):
    """Return padded expanded spans and visual-token counts for each sample."""
    spans = []
    visual_counts = []
    for batch_index in range(input_ids.shape[0]):
        image_positions = (
            input_ids[batch_index] == image_token_index
        ).nonzero(as_tuple=False)
        if image_positions.shape[0] != 1:
            raise ValueError(
                "GeoGate batch requires exactly one image token per sample"
            )
        original_valid = (
            int(input_attention_mask[batch_index].sum().item())
            if input_attention_mask is not None
            else int(input_ids.shape[1])
        )
        expanded_valid = (
            int(expanded_attention_mask[batch_index].sum().item())
            if expanded_attention_mask is not None
            else int(expanded_width)
        )
        image_pos = int(image_positions[0, 0].item())
        if input_attention_mask is None:
            image_pos_unpadded = image_pos
        else:
            image_pos_unpadded = int(
                input_attention_mask[batch_index, :image_pos].sum().item()
            )
        visual_count = expanded_valid - (original_valid - 1)
        if visual_count <= 0:
            raise ValueError("Expanded multimodal input has no visual tokens")
        left_pad = int(expanded_width) - expanded_valid
        span_start = left_pad + image_pos_unpadded
        spans.append((span_start, span_start + visual_count))
        visual_counts.append(visual_count)
    return spans, visual_counts


def build_keep_indices(scores, spans, sequence_length, keep_ratio):
    """Build one fixed-width kept-index row per sample."""
    counts = [sp1 - sp0 for sp0, sp1 in spans]
    if len(set(counts)) != 1:
        raise ValueError(
            "GeoGate batch requires equal spatial-token counts; "
            "bucket questions by scene or exact visual-token count"
        )
    visual_count = counts[0]
    keep_count = max(1, int(round(float(keep_ratio) * visual_count)))
    rows = []
    for batch_index, (sp0, sp1) in enumerate(spans):
        spatial_scores = scores[batch_index, sp0:sp1]
        keep_spatial = torch.sort(
            torch.topk(spatial_scores, min(keep_count, visual_count)).indices
        ).values + sp0
        rows.append(torch.cat([
            torch.arange(0, sp0, device=scores.device),
            keep_spatial,
            torch.arange(sp1, sequence_length, device=scores.device),
        ]))
    keep_indices = torch.stack(rows, dim=0)
    return keep_indices, keep_count


def gather_sequence(values, indices):
    """Gather a different sequence-index row for every batch element."""
    if values.ndim == 2:
        return torch.gather(values, 1, indices)
    if values.ndim == 3:
        expanded = indices.unsqueeze(-1).expand(-1, -1, values.shape[-1])
        return torch.gather(values, 1, expanded)
    raise ValueError(f"Unsupported sequence tensor rank: {values.ndim}")


def gather_cache(values, indices):
    """Gather batched KV tensors shaped (B, H, T, D)."""
    expanded = indices[:, None, :, None].expand(
        -1,
        values.shape[1],
        -1,
        values.shape[3],
    )
    return torch.gather(values, 2, expanded)


def enable_auto_spans(model, image_token_index):
    """Infer one expanded spatial span per sample during multimodal prefill."""
    if getattr(model, "_gg_batch_auto_spans_enabled", False):
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
            language_model._gg_batch_spans = spans
            language_model._gg_batch_visual_counts = visual_counts
        return result

    model.prepare_inputs_labels_for_multimodal = types.MethodType(
        prepare_with_spans,
        model,
    )
    model._gg_batch_auto_spans_enabled = True
    return model


def enable_geogate_batch(model, prune_layer=14, keep_ratio=0.10):
    """Patch fixed single-drop GeoGate for scene-bucketed tensor batches."""
    language_model = model.model
    language_model._gg_batch_prune_layer = int(prune_layer)
    language_model._gg_batch_keep_ratio = float(keep_ratio)
    language_model._gg_batch_spans = None
    language_model._gg_batch_visual_counts = None
    language_model._gg_batch_next_pos = None
    language_model._gg_batch_cache_mask = None
    language_model._gg_batch_trace = []

    for layer in language_model.layers:
        rotary = layer.self_attn.rotary_emb
        if getattr(rotary, "_gg_batch_wrapped", False):
            continue
        original_rotary = rotary.forward

        def wrapped(x, seq_len=None, _original=original_rotary, _lm=language_model):
            need = max(seq_len or 0, getattr(_lm, "_gg_batch_maxpos", 0))
            return _original(x, seq_len=need)

        rotary.forward = wrapped
        rotary._gg_batch_wrapped = True

    language_model.forward = types.MethodType(
        _geogate_batch_forward,
        language_model,
    )

    if not getattr(model, "_gg_batch_generation_wrapped", False):
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
                and language_model._gg_batch_next_pos is not None
            ):
                prepared_ids = output.get("input_ids")
                if prepared_ids is not None and prepared_ids.shape[1] > 1:
                    output["input_ids"] = prepared_ids[:, -1:]
                if output.get("position_ids") is not None:
                    output["position_ids"] = None
            return output

        model.prepare_inputs_for_generation = prepare_generation
        model._gg_batch_generation_wrapped = True
    return model


def _geogate_batch_forward(
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
    """Transformers 4.37 LlamaModel forward with one batched pruning event."""
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

    spans = self._gg_batch_spans
    prune_prefill = is_prefill and spans is not None
    if is_prefill:
        self._gg_batch_trace = []
        self._gg_batch_maxpos = int(position_ids.max().item()) + 641
        self._gg_batch_next_pos = position_ids.max(dim=1).values + 1
        self._gg_batch_cache_mask = None

    if not is_prefill and use_cache:
        real_past = past_key_values.get_seq_length()
        past_length = real_past
        if self._gg_batch_cache_mask is None:
            raise RuntimeError("Missing pruned cache mask during batch decode")
        current_mask = torch.ones(
            (batch, sequence_length),
            dtype=self._gg_batch_cache_mask.dtype,
            device=inputs_embeds.device,
        )
        attention_mask = torch.cat(
            [self._gg_batch_cache_mask, current_mask],
            dim=1,
        )
        position_ids = self._gg_batch_next_pos[:, None] + torch.arange(
            sequence_length,
            device=inputs_embeds.device,
        )[None, :]
        self._gg_batch_next_pos = (
            self._gg_batch_next_pos + sequence_length
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
                and layer_index == self._gg_batch_prune_layer
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
            and layer_index == self._gg_batch_prune_layer
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
            keep_indices, keep_count = build_keep_indices(
                scores,
                spans,
                sequence_length,
                self._gg_batch_keep_ratio,
            )
            hidden_states = gather_sequence(hidden_states, keep_indices)
            position_ids = gather_sequence(position_ids, keep_indices)
            attention_mask = gather_sequence(attention_mask, keep_indices)
            sequence_length = hidden_states.shape[1]
            attention_mask_4d = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch, sequence_length),
                hidden_states,
                0,
            )
            self._gg_batch_trace.append({
                "event": "prune",
                "layer": layer_index,
                "batch": batch,
                "before": pre_prune_length,
                "after": sequence_length,
                "visual_counts": list(self._gg_batch_visual_counts),
                "keep_count": keep_count,
                "valid_after": attention_mask.sum(dim=1).tolist(),
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
        self._gg_batch_cache_mask = attention_mask
        self._gg_batch_spans = None
    elif not is_prefill and use_cache:
        self._gg_batch_cache_mask = attention_mask

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
