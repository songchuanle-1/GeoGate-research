import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from analysis.score_openeqa_hm3d_v14 import (
    exact_and_f1,
    summarize,
)


class ScoreOpenEQAHM3DV14Tests(unittest.TestCase):
    def test_lexical_metrics_use_best_reference(self):
        exact, f1 = exact_and_f1(
            "a brown chair",
            ["chair", "a brown chair"],
        )
        self.assertEqual(exact, 1)
        self.assertEqual(f1, 1.0)

    def test_structural_authorization_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.jsonl"
            raw.write_text("", encoding="utf-8")
            dataset = root / "dataset.json"
            dataset.write_text("[]", encoding="utf-8")
            audit = root / "audit.json"
            audit.write_text(json.dumps({
                "state": "passed",
                "scoring_authorized": False,
            }), encoding="utf-8")
            grades = root / "grades.json"
            grades.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                summarize(
                    raw,
                    dataset,
                    audit,
                    grades,
                    grades,
                )

    def test_paired_summary_applies_frozen_margin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_rows = []
            raw_rows = []
            native_grades = {}
            geogate_grades = {}
            for index in range(4):
                question_id = f"q{index}"
                episode = f"hm3d-v0/00{index // 2}-hm3d-Scene{index // 2}"
                dataset_rows.append({
                    "question_id": question_id,
                    "question": "What is it?",
                    "answer": "chair",
                    "category": "object recognition",
                    "episode_history": episode,
                })
                for condition in ("native", "geogate_L14_R010"):
                    raw_rows.append({
                        "question_id": question_id,
                        "condition": condition,
                        "answer": "chair",
                    })
                native_grades[question_id] = 5
                geogate_grades[question_id] = 5

            raw = root / "raw.jsonl"
            raw.write_text(
                "".join(json.dumps(row) + "\n" for row in raw_rows),
                encoding="utf-8",
            )
            dataset = root / "dataset.json"
            dataset.write_text(
                json.dumps(dataset_rows),
                encoding="utf-8",
            )
            native_path = root / "native.json"
            native_path.write_text(
                json.dumps(native_grades),
                encoding="utf-8",
            )
            geogate_path = root / "geogate.json"
            geogate_path.write_text(
                json.dumps(geogate_grades),
                encoding="utf-8",
            )
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps({
                "state": "passed",
                "scoring_authorized": True,
                "raw": {
                    "sha256": hashlib.sha256(
                        raw.read_bytes()
                    ).hexdigest(),
                },
            }), encoding="utf-8")

            result = summarize(
                raw,
                dataset,
                audit_path,
                native_path,
                geogate_path,
            )
            self.assertEqual(
                result["decision"],
                "supports_frozen_noninferior_transfer",
            )
            self.assertEqual(
                result["official_llm_match"]["paired_delta"]["ci"],
                [0.0, 0.0],
            )


if __name__ == "__main__":
    unittest.main()
