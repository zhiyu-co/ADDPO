#!/usr/bin/env python3
"""Extract mirrored MMAU audio while retaining the official benchmark labels."""

import argparse
import json
import re
from pathlib import Path

import pyarrow.parquet as pq


parser = argparse.ArgumentParser()
parser.add_argument("--official-json", required=True, type=Path)
parser.add_argument("--parquet-dir", required=True, type=Path)
parser.add_argument("--audio-dir", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()

official = json.loads(args.official_json.read_text(encoding="utf-8"))
official_by_id = {row["id"]: row for row in official}
if len(official) != 1000 or len(official_by_id) != 1000:
    raise ValueError(f"Expected 1000 unique official rows, got {len(official_by_id)}")

args.audio_dir.mkdir(parents=True, exist_ok=True)
seen = set()
audio_paths = {}
checked_fields = ("question", "answer", "dataset", "task", "split", "category", "sub-category", "difficulty")

shards = sorted(args.parquet_dir.glob("*.parquet"))
if not shards:
    raise ValueError(f"No Parquet files found in {args.parquet_dir}")

for shard in shards:
    for mirrored in pq.read_table(shard).to_pylist():
        if "other_attributes" in mirrored:
            metadata = json.loads(mirrored["other_attributes"])
            sample_id = metadata["id"]
            mirror_question = mirrored["instruction"]
            mirror_choices = [re.sub(r"^\([A-Z]\)\s*", "", choice).strip() for choice in mirrored["choices"]]
            mirror_answer = re.sub(r"^\([A-Z]\)\s*", "", mirrored["answer"]).strip()
            audio = mirrored["context"]["bytes"]
        else:
            metadata = mirrored
            sample_id = mirrored["id"]
            mirror_question = mirrored["question"]
            mirror_choices = json.loads(mirrored["choices"])
            mirror_answer = mirrored["answer"]
            audio = mirrored["audio"]["bytes"]
        if sample_id not in official_by_id:
            raise ValueError(f"Mirror contains unknown UUID {sample_id}")
        if sample_id in seen:
            raise ValueError(f"Mirror contains duplicate UUID {sample_id}")
        source = official_by_id[sample_id]
        for field in checked_fields:
            mirror_value = mirror_question if field == "question" else mirror_answer if field == "answer" else metadata[field]
            source_value = source[field].strip() if field == "answer" else source[field]
            if mirror_value != source_value:
                raise ValueError(f"Metadata mismatch for {sample_id}: {field}")
        if mirror_choices != [choice.strip() for choice in source["choices"]]:
            raise ValueError(f"Choice mismatch for {sample_id}")

        if audio[:4] == b"RIFF":
            suffix = ".wav"
        elif audio[:3] == b"ID3" or audio[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
            suffix = ".mp3"
        else:
            raise ValueError(f"Unsupported audio payload for {sample_id}: {audio[:8].hex()}")
        target = args.audio_dir / f"{sample_id}{suffix}"
        temporary = target.with_suffix(target.suffix + ".partial")
        temporary.write_bytes(audio)
        temporary.replace(target)
        audio_paths[sample_id] = target.resolve()
        seen.add(sample_id)

if seen != set(official_by_id):
    missing = sorted(set(official_by_id) - seen)
    raise ValueError(f"Mirror is missing {len(missing)} official UUIDs: {missing[:5]}")

prepared = []
for source in official:
    row = dict(source)
    row["audio"] = str(audio_paths[source["id"]])
    prepared.append(row)

args.output.write_text(json.dumps(prepared, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Prepared and metadata-verified {len(prepared)} MMAU test-mini samples at {args.output}")
