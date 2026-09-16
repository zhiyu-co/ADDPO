#!/usr/bin/env python3
"""Build and sample MusicBench key/tempo/time-signature MCQ rows."""

import argparse
import json
import random
import re
from pathlib import Path


NOTES = ["C", "D", "Db", "E", "Eb", "F", "F#", "G", "Gb", "A", "Ab", "Bb", "B"]
EQUIVALENTS = {"B#": "C", "C#": "Db", "D#": "Eb", "Fb": "E", "E#": "F", "G#": "Ab", "A#": "Bb", "Cb": "B"}
TERMS = [
    ("Larghissimo", 0, 25), ("Grave", 25, 45), ("Largo", 40, 60),
    ("Lento", 45, 60), ("Larghetto", 60, 66), ("Adagio", 66, 76),
    ("Adagietto", 72, 76), ("Andante", 76, 108), ("Andantino", 80, 108),
    ("Andante moderato", 92, 112), ("Moderato", 108, 120),
    ("Allegretto", 112, 120), ("Allegro moderato", 116, 120),
    ("Allegro", 120, 168), ("Vivace", 140, 176),
    ("Vivacissimo", 172, 176), ("Allegrissimo", 172, 176),
    ("Presto", 168, 200), ("Prestissimo", 200, float("inf")),
]
TIME_SIGNATURES = ["4/4", "3/4", "2/4", "6/8", "12/8", "5/4", "7/8", "9/8"]


def choices(correct: str, pool: list[str], rng: random.Random) -> tuple[list[str], int]:
    options = [correct] + rng.sample([x for x in pool if x.lower() != correct.lower()], 3)
    rng.shuffle(options)
    return options, options.index(correct)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-missing", action="store_true", help="Select rows before extracting their audio")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    candidates = []
    term_names = [x[0] for x in TERMS]
    term_pattern = re.compile(r"\b(" + "|".join(map(re.escape, term_names)) + r")\b", re.I)

    with args.metadata.open(encoding="utf-8") as reader:
        for line in reader:
            source = json.loads(line)
            audio = (args.audio_root / source["location"]).resolve()
            stem = Path(source["location"]).stem

            key_name, mode = source["key"]
            key_name = EQUIVALENTS.get(key_name, key_name)
            correct = f"{key_name} {mode.lower()}"
            pool = [f"{note} {mode.lower()}" for note in NOTES]
            opts, answer = choices(correct, pool, rng)
            candidates.append({"id": f"key_{stem}", "question_text": "Which of the following keys best fits this piece?", "multi_choice": opts, "answer": answer, "audio_path": str(audio), "dataset_name": "MusicBench"})

            match = term_pattern.search(source.get("prompt_bpm", ""))
            if match:
                correct = match.group(1)
                bounds = next((lo, hi) for term, lo, hi in TERMS if term.lower() == correct.lower())
                pool = [term for term, lo, hi in TERMS if not (lo <= bounds[1] and hi >= bounds[0])]
                if len(pool) >= 3:
                    opts, answer = choices(correct, pool, rng)
                    candidates.append({"id": f"tempo_{stem}", "question_text": "Which of the following tempo marks best fits this piece?", "multi_choice": opts, "answer": answer, "audio_path": str(audio), "dataset_name": "MusicBench"})

            match = re.search(r"(\d+/\d+)", source.get("prompt_bt", ""))
            if match and match.group(1) in TIME_SIGNATURES:
                correct = match.group(1)
                pool = [x for x in TIME_SIGNATURES if not (correct in {"3/4", "6/8"} and x in {"3/4", "6/8"})]
                opts, answer = choices(correct, pool, rng)
                candidates.append({"id": f"time_{stem}", "question_text": "Which of the following time signatures best fits this piece?", "multi_choice": opts, "answer": answer, "audio_path": str(audio), "dataset_name": "MusicBench"})

    available = candidates if args.allow_missing else [x for x in candidates if Path(x["audio_path"]).is_file()]
    if len(available) < args.count:
        raise ValueError(f"Only {len(available)} prepared rows have audio; requested {args.count}")
    selected = rng.sample(available, args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as writer:
        for row in selected:
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(selected)} rows from {len({x['audio_path'] for x in selected})} audio files to {args.output}")


if __name__ == "__main__":
    main()
