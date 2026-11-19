"""One-preparation cache shared by native and intervention generations."""

import types


class PreparedGenerationCache:
    def __init__(self, model):
        self.model = model
        self.original = model.prepare_inputs_labels_for_multimodal
        self.prepared = None
        self.input_ids = None
        self.prefill_hits = 0

        def wrapped(_model, *args, **kwargs):
            import torch

            ids = args[0] if args else kwargs.get("input_ids")
            images = args[5] if len(args) > 5 else kwargs.get("images")
            is_prefill = (
                torch.is_tensor(ids)
                and ids.ndim == 2
                and ids.shape[1] > 1
                and images is not None
            )
            if not is_prefill:
                return self.original(*args, **kwargs)
            if self.prepared is None or self.input_ids is None:
                raise RuntimeError("Prepared generation cache is not initialized")
            if (
                ids.shape != self.input_ids.shape
                or not torch.equal(ids, self.input_ids)
            ):
                raise RuntimeError("Cached prompt token IDs differ")
            self.prefill_hits += 1
            return self.prepared

        self.wrapper = types.MethodType(wrapped, model)
        model.prepare_inputs_labels_for_multimodal = self.wrapper

    def prepare(self, input_ids, *args, **kwargs):
        if self.prepared is not None:
            raise RuntimeError("Clear the previous prepared input first")
        prepared = self.original(input_ids, *args, **kwargs)
        if len(prepared) < 5 or prepared[4] is None:
            raise RuntimeError("Multimodal preparation returned no embedding")
        self.prepared = prepared
        self.input_ids = input_ids.clone()
        self.prefill_hits = 0
        return prepared

    def clear(self):
        self.prepared = None
        self.input_ids = None
        self.prefill_hits = 0

    def close(self):
        if self.model.prepare_inputs_labels_for_multimodal is not self.wrapper:
            raise RuntimeError("Prepared generation cache hook changed")
        self.model.prepare_inputs_labels_for_multimodal = self.original
        self.clear()
