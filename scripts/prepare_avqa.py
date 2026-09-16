#!/usr/bin/env python3
"""Download AVQA audio and create a deterministic training subset."""

import argparse
import concurrent.futures
import json
import random
import subprocess
import tempfile
import time
import urllib.parse
from pathlib import Path

from huggingface_hub import list_repo_files
from tqdm import tqdm


BASE_URL = "https://huggingface.co/datasets/juyil/AVQA-videos/resolve/main/videos/"


def convert_one(video_name: str, audio_dir: Path, ffmpeg: str) -> Path:
    output = audio_dir / f"{video_name}.wav"
    if output.is_file() and output.stat().st_size > 0:
        return output
    url = BASE_URL + urllib.parse.quote(f"{video_name}.mp4")
    with tempfile.NamedTemporaryFile(dir=audio_dir, suffix=".mp4") as tmp:
        subprocess.run(
            ["curl", "-L", "--fail", "--silent", "--show-error", "--retry", "10",
             "--retry-all-errors", "--retry-delay", "2", "-o", tmp.name, url],
            check=True,
        )
        tmp.flush()
        subprocess.run(
            [ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-i", tmp.name,
             "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output)],
            check=True,
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--buffer", type=int, default=1200, help="Extra rows to cover unavailable or transiently failed source clips")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()

    samples = json.loads(args.annotations.read_text())
    for attempt in range(10):
        try:
            repo_files = list_repo_files("juyil/AVQA-videos", repo_type="dataset")
            break
        except Exception:
            if attempt == 9:
                raise
            time.sleep(2 ** min(attempt, 5))
    available_names = {
        Path(path).stem for path in repo_files
        if path.startswith("videos/") and path.endswith(".mp4")
    }
    eligible = [
        x for x in samples
        if x.get("question_relation") != "View"
        and "color" not in x.get("question_text", "").lower()
        and x.get("video_name") in available_names
    ]
    if len(eligible) < args.count:
        raise ValueError(f"Only {len(eligible)} eligible AVQA rows for requested {args.count}")
    random.Random(args.seed).shuffle(eligible)
    candidates = eligible[: args.count + args.buffer]
    names = sorted({x["video_name"] for x in candidates})
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(convert_one, name, args.audio_dir, args.ffmpeg) for name in names]
        failures = 0
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="AVQA audio"):
            try:
                future.result()
            except Exception:
                failures += 1

    selected = [x for x in candidates if (args.audio_dir / f"{x['video_name']}.wav").is_file()][: args.count]
    if len(selected) < args.count:
        raise RuntimeError(f"Only {len(selected)} usable rows after {failures} unavailable/corrupt clips; increase --buffer")

    with args.output.open("w", encoding="utf-8") as writer:
        for sample in selected:
            row = dict(sample)
            row["audio_path"] = str((args.audio_dir / f"{row['video_name']}.wav").resolve())
            row["dataset_name"] = "AVQA"
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(selected)} rows ({len(names)} unique audio files) to {args.output}")


if __name__ == "__main__":
    main()
