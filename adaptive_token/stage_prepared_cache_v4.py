"""Explicit per-question preparation reuse for the eight-field 3D-LLaVA API."""

import types


class StagePreparedCache:
    def __init__(self, model):
        self.model = model
        self.original = model.prepare_inputs_labels_for_multimodal
        self.prepared = None
        self.input_ids = None
        self.hits = 0

        def wrapped(_model, *args, **kwargs):
            ids = args[0] if args else kwargs.get("input_ids")
            if ids is None or ids.shape[1] == 1:
                return self.original(*args, **kwargs)
            if self.prepared is None:
                raise RuntimeError("Question preparation cache not initialized")
            if ids.shape != self.input_ids.shape or not bool((ids == self.input_ids).all()):
                raise ValueError("Cached question token IDs differ")
            self.hits += 1
            return self.prepared

        self.wrapper = types.MethodType(wrapped, model)
        model.prepare_inputs_labels_for_multimodal = self.wrapper

    def prepare(self, input_ids, **kwargs):
        if self.prepared is not None:
            raise RuntimeError("Clear previous question before preparing another")
        prepared = self.original(input_ids, None, None, None, None, **kwargs)
        if len(prepared) != 8 or prepared[4] is None:
            raise ValueError("Expected eight-field 3D-LLaVA multimodal preparation")
        self.input_ids = input_ids.clone()
        self.prepared = prepared
        self.hits = 0
        return prepared

    def clear(self):
        self.prepared = None
        self.input_ids = None
        self.hits = 0

    def close(self):
        if self.model.prepare_inputs_labels_for_multimodal is not self.wrapper:
            raise RuntimeError("Preparation hook changed externally")
        self.model.prepare_inputs_labels_for_multimodal = self.original
        self.clear()
