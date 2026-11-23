import json
import copy
from pathlib import Path
import tempfile
import unittest

from analysis.audit_openeqa_hm3d_v14 import (
    FROZEN_DATASET_SHA256,
    audit,
)


def fake_hash(character):
    return character * 64


class AuditOpenEQAHM3DV14Tests(unittest.TestCase):
    def write_fixture(self, root, alter=None):
        manifest = {
            "is_full": False,
            "question_ids": ["q1"],
            "dataset": {"sha256": FROZEN_DATASET_SHA256},
            "protocol": {
                "seed": 20260913,
                "conditions": ["native", "geogate_L14_R010"],
                "condition_order": "native_then_geogate",
                "decoding": {
                    "do_sample": False,
                    "max_new_tokens": 512,
                    "num_beams": 1,
                },
            },
            "provenance": {
                "runner_sha256": fake_hash("a"),
                "hm3d_adapter_sha256": fake_hash("b"),
                "preflight_sha256": fake_hash("7"),
                "audit_sha256": fake_hash("8"),
                "scorer_sha256": fake_hash("9"),
                "geogate_sha256": fake_hash("c"),
                "scene_cap_sha256": fake_hash("d"),
            },
        }
        shared = {
            "question_id": "q1",
            "episode": "hm3d-v0/000-hm3d-SceneA",
            "scene_id": "SceneA",
            "category": "spatial",
            "question": "Where?",
            "prompt": "<image>\nWhere?",
            "frame_selection": {
                "indices": [0, 1],
                "selection_sha256": fake_hash("e"),
            },
            "input_tensor_sha256": {
                "images": fake_hash("f"),
                "depths": fake_hash("1"),
                "poses": fake_hash("2"),
                "intrinsics": fake_hash("3"),
            },
            "expanded_input_sha256": fake_hash("6"),
            "full_tokens": 130,
            "spatial_start": 12,
            "input_visual_tokens": 100,
            "seed": 20260913,
            "deterministic_scene_cap": [],
            "preparation_calls": 1,
            "prepared_prefill_cache_hits": 2,
            "latency_s": 1.0,
        }
        rows = [
            {
                **copy.deepcopy(shared),
                "condition": "native",
                "answer": "left",
                "output_ids_sha256": fake_hash("4"),
                "forward_path": "native_unpatched_llama_forward",
            },
            {
                **copy.deepcopy(shared),
                "condition": "geogate_L14_R010",
                "answer": "left",
                "output_ids_sha256": fake_hash("5"),
                "forward_path": "geogate_patched_llama_forward",
            },
        ]
        if alter is not None:
            alter(manifest, rows)
        manifest_path = root / "manifest.json"
        raw_path = root / "raw.jsonl"
        manifest_path.write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        raw_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        return raw_path, manifest_path

    def test_valid_smoke_passes_but_does_not_authorize_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            raw, manifest = self.write_fixture(Path(directory))
            result = audit(raw, manifest)
            self.assertEqual(result["state"], "passed")
            self.assertFalse(result["scoring_authorized"])

    def test_paired_input_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            def alter(manifest, rows):
                rows[1]["input_tensor_sha256"]["images"] = fake_hash("9")

            raw, manifest = self.write_fixture(
                Path(directory),
                alter=alter,
            )
            result = audit(raw, manifest)
            self.assertEqual(result["state"], "failed")
            self.assertIn(
                "q1:paired_input_mismatch",
                result["errors"],
            )

    def test_condition_order_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            def alter(manifest, rows):
                rows.reverse()

            raw, manifest = self.write_fixture(
                Path(directory),
                alter=alter,
            )
            result = audit(raw, manifest)
            self.assertEqual(result["state"], "failed")
            self.assertIn(
                "q1:conditions_or_order_mismatch",
                result["errors"],
            )


if __name__ == "__main__":
    unittest.main()
