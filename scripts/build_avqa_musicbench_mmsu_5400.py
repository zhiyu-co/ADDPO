#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def unique_by_id(rows: list[dict]) -> list[dict]:
    unique = {}
    for row in rows:
        unique.setdefault(row["id"], row)
    return list(unique.values())


def stratified_sample(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Proportionally sample all task_name strata with deterministic largest remainders."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["task_name"]].append(row)

    exact = {key: count * len(group) / len(rows) for key, group in groups.items()}
    allocation = {key: int(value) for key, value in exact.items()}
    remaining = count - sum(allocation.values())
    order = sorted(groups, key=lambda key: (-(exact[key] - allocation[key]), key))
    for key in order[:remaining]:
        allocation[key] += 1

    sampled = []
    for index, key in enumerate(sorted(groups)):
        sampled.extend(random.Random(seed + index).sample(groups[key], allocation[key]))
    return sampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--avqa", type=Path, required=True)
    parser.add_argument("--musicbench", type=Path, required=True)
    parser.add_argument("--mmsu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mmsu-max-seconds",
        type=float,
        default=0.0,
        help="Optional duration cap; non-positive values keep all MMSU clips",
    )
    args = parser.parse_args()

    avqa = random.Random(args.seed).sample(unique_by_id(read_jsonl(args.avqa)), args.count)
    musicbench = random.Random(args.seed + 1).sample(
        unique_by_id(read_jsonl(args.musicbench)), args.count
    )
    mmsu_all = json.loads(args.mmsu.read_text(encoding="utf-8"))
    if args.mmsu_max_seconds > 0:
        mmsu_all = [
            row
            for row in mmsu_all
            if sf.info(row["audio_path"]).frames / sf.info(row["audio_path"]).samplerate
            <= args.mmsu_max_seconds
        ]
    if len(mmsu_all) < args.count:
        raise ValueError(f"Only {len(mmsu_all)} MMSU rows satisfy the duration limit")
    mmsu_rows = stratified_sample(mmsu_all, args.count, args.seed + 2)

    converted_mmsu = []
    for row in mmsu_rows:
        choices = [row[f"choice_{letter}"] for letter in "abcd"]
        matches = [index for index, choice in enumerate(choices) if choice == row["answer_gt"]]
        if len(matches) != 1:
            raise ValueError(f"Expected one matching answer for {row['id']}, got {matches}")
        audio_path = Path(row["audio_path"]).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        converted_mmsu.append({
            "id": f"mmsu_{row['id']}",
            "question_text": row["question"],
            "multi_choice": choices,
            "answer": matches[0],
            "audio_path": str(audio_path),
            "dataset_name": "MMSU",
            "modality": "speech",
            "task_name": row["task_name"],
            "category": row["category"],
            "sub_category": row["sub-category"],
            "sub_sub_category": row["sub-sub-category"],
            "linguistics_sub_discipline": row["linguistics_sub_discipline"],
            "source_id": row["id"],
        })

    rows = avqa + musicbench + converted_mmsu
    random.Random(args.seed + 3).shuffle(rows)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)

    print(f"rows={len(rows)}")
    print("sources=" + json.dumps(Counter(row["dataset_name"] for row in rows), sort_keys=True))
    print("mmsu_categories=" + json.dumps(Counter(row["category"] for row in converted_mmsu), sort_keys=True))
    print(f"mmsu_tasks={len(set(row['task_name'] for row in converted_mmsu))}")
    print(f"mmsu_max_seconds={args.mmsu_max_seconds}")
    print("sha256=" + hashlib.sha256(payload).hexdigest())


if __name__ == "__main__":
    main()
