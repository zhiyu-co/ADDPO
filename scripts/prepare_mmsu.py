import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description="Extract MMSU parquet audio and metadata")
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--metadata-file", required=True)
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for parquet_path in sorted(Path(args.parquet_dir).glob("*.parquet")):
        for row in pq.read_table(parquet_path).to_pylist():
            audio = row.pop("audio")
            suffix = Path(audio.get("path") or "audio.wav").suffix or ".wav"
            audio_path = audio_dir / f"{row['id']}{suffix}"
            if not audio_path.exists() or audio_path.stat().st_size != len(audio["bytes"]):
                audio_path.write_bytes(audio["bytes"])
            row["audio_path"] = str(audio_path)
            records.append(row)

    with open(args.metadata_file, "w", encoding="utf-8") as writer:
        json.dump(records, writer, ensure_ascii=False)
    print(f"Prepared {len(records)} samples in {audio_dir}")


if __name__ == "__main__":
    main()
