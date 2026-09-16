"""Dataset and prompt construction shared by GRPO and AD²PO."""

import json
import logging
from pathlib import Path

from torch.utils.data import Dataset


def build_example(row: dict, think_max_len: int = 128) -> dict:
    required = {"question_text", "multi_choice", "answer", "audio_path"}
    missing = required - row.keys()
    if missing:
        raise ValueError(f"Training row is missing required fields: {sorted(missing)}")
    if not isinstance(row["multi_choice"], list) or len(row["multi_choice"]) < 2:
        raise ValueError(f"Invalid choices for sample {row.get('id', '<unknown>')}")
    if not isinstance(row["answer"], int) or not 0 <= row["answer"] < len(row["multi_choice"]):
        raise ValueError(f"Invalid answer index for sample {row.get('id', '<unknown>')}")

    question = row["question_text"].replace("video", "audio")
    prompt = (
        f"{question} Please choose the answer from the following options: "
        f"{row['multi_choice']}. Output the thinking process (less than {think_max_len} words) "
        "in <think> </think> and final answer in <answer> </answer>. "
        "Do not output anything after </answer>."
    )
    return {
        **row,
        "prompt": [{
            "role": "user",
            "content": [
                {"type": "audio", "audio": row["audio_path"]},
                {"type": "text", "text": prompt},
            ],
        }],
        "solution": f"<answer>{row['multi_choice'][row['answer']]}</answer>",
    }


class AudioDataset(Dataset):
    def __init__(self, data_file: str, is_think: bool = True, think_max_len: int = 128, **_) -> None:
        if not is_think:
            raise ValueError("The released training recipe requires the reasoning prompt")
        self.path = Path(data_file)
        with self.path.open(encoding="utf-8") as stream:
            self.rows = [json.loads(line) for line in stream if line.strip()]
        self.think_max_len = think_max_len
        logging.info("Loaded %d training rows from %s", len(self.rows), self.path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return build_example(dict(self.rows[index]), self.think_max_len)
