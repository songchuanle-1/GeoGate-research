#!/usr/bin/env python3
"""Merge, audit, and score four frozen v15 full-run shards."""

import argparse
import hashlib
import json
from pathlib import Path

from analysis.audit_readout_generation_v15 import audit_rows
from analysis.score_readout_generation_v15 import load_gt, summarize


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json_exclusive(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--raw", type=Path, nargs=4, required=True)
    parser.add_argument("--manifest", type=Path, nargs=4, required=True)
    parser.add_argument("--status", type=Path, nargs=4, required=True)
    parser.add_argument("--audit", type=Path, nargs=4, required=True)
    parser.add_argument("--merged-raw", type=Path, required=True)
    parser.add_argument("--merged-manifest", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--score-out", type=Path, required=True)
    args = parser.parse_args()

    for path in (
        args.merged_raw,
        args.merged_manifest,
        args.audit_out,
        args.score_out,
    ):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    questions = read_json(args.questions)
    official_ids = [str(row["question_id"]) for row in questions]
    if len(official_ids) != 4675 or len(set(official_ids)) != 4675:
        raise ValueError("Expected 4675 unique official questions")

    manifests = [read_json(path) for path in args.manifest]
    statuses = [read_json(path) for path in args.status]
    audits = [read_json(path) for path in args.audit]
    rows_by_question = {}
    shard_summaries = []
    reference = None
    all_scenes = set()
    for index, (
        raw_path,
        manifest_path,
        status,
        audit,
    ) in enumerate(zip(args.raw, args.manifest, statuses, audits)):
        manifest = manifests[index]
        if (
            manifest["shard"]["index"] != index
            or manifest["shard"]["count"] != 4
            or status["state"] != "finished"
            or status["raw_sha256"] != digest_file(raw_path)
            or status["manifest_sha256"] != digest_file(manifest_path)
            or status["repeat_verification"]
            or status["rows"] != status["questions"] * 16
            or audit["state"] != "passed"
            or audit["raw_sha256"] != digest_file(raw_path)
            or audit["manifest_sha256"] != digest_file(manifest_path)
        ):
            raise ValueError(f"Invalid v15 shard {index}")
        frozen = {
            key: manifest[key]
            for key in ("model", "questions", "inventory", "protocol")
        }
        frozen["questions"] = {
            key: frozen["questions"][key]
            for key in ("path", "sha256", "source_count")
        }
        frozen["protocol"] = dict(frozen["protocol"])
        frozen["protocol"]["repeat_verification"] = False
        if reference is None:
            reference = frozen
        elif frozen != reference:
            raise ValueError("Frozen v15 protocol differs across shards")

        shard_rows = read_jsonl(raw_path)
        audit_rows(shard_rows, manifest["question_ids"])
        if len(shard_rows) != status["rows"]:
            raise ValueError(f"v15 shard {index} row count mismatch")
        scenes = set(manifest["scenes"])
        if all_scenes.intersection(scenes):
            raise ValueError("v15 scene assigned to multiple shards")
        all_scenes.update(scenes)
        for row in shard_rows:
            question_id = str(row["question_id"])
            rows_by_question.setdefault(question_id, []).append(row)
        shard_summaries.append({
            "index": index,
            "questions": status["questions"],
            "scenes": status["scenes"],
            "rows": status["rows"],
            "raw_sha256": status["raw_sha256"],
            "manifest_sha256": status["manifest_sha256"],
            "audit_sha256": digest_file(args.audit[index]),
            "estimated_attention_weight": manifest["shard"][
                "estimated_attention_weight"
            ],
        })

    if (
        set(rows_by_question) != set(official_ids)
        or len(all_scenes) != 71
        or any(len(rows_by_question[value]) != 16 for value in official_ids)
    ):
        raise ValueError("Four v15 shards do not form the frozen full set")
    merged_rows = [
        row
        for question_id in official_ids
        for row in rows_by_question[question_id]
    ]
    args.merged_raw.parent.mkdir(parents=True, exist_ok=True)
    with args.merged_raw.open("x", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    merged_manifest = {
        "created_from_shards": shard_summaries,
        "is_full": True,
        "question_ids": official_ids,
        "scenes": sorted(all_scenes),
        "model": reference["model"],
        "questions": reference["questions"],
        "inventory": reference["inventory"],
        "protocol": reference["protocol"],
        "provenance": {
            "finalizer_sha256": digest_file(__file__),
            "merged_raw_sha256": digest_file(args.merged_raw),
        },
    }
    write_json_exclusive(args.merged_manifest, merged_manifest)
    audit = audit_rows(merged_rows, official_ids)
    audit["merged_raw_sha256"] = digest_file(args.merged_raw)
    audit["merged_manifest_sha256"] = digest_file(args.merged_manifest)
    write_json_exclusive(args.audit_out, audit)
    score = summarize(merged_rows, official_ids, load_gt(args.gt))
    score["hashes"] = {
        "merged_raw": digest_file(args.merged_raw),
        "merged_manifest": digest_file(args.merged_manifest),
        "gt": digest_file(args.gt),
        "finalizer": digest_file(__file__),
    }
    write_json_exclusive(args.score_out, score)
    print(json.dumps({
        "state": "finished",
        "questions": len(official_ids),
        "scenes": len(all_scenes),
        "rows": len(merged_rows),
        "decision": score["decision"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
