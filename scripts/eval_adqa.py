#!/usr/bin/env python3
"""Run resumable Qwen2.5-Omni inference on ADQA-Bench."""

import argparse
import json
import os
import re
from pathlib import Path

import torch
from qwen_omni_utils import process_mm_info
from tqdm import tqdm
from transformers import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
    StoppingCriteriaList,
)

from utils.generation import DecodedStringStoppingCriteria


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-file", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--think-max-len", type=int, default=128)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def normalize(value: str) -> str:
    return " ".join(re.findall(r"\w+", str(value).casefold()))


def extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    return (match.group(1) if match else text).strip()


def answer_letter(answer: str, choices: list[str]) -> str:
    labels = "".join(chr(ord("A") + index) for index in range(len(choices)))
    match = re.fullmatch(rf"\s*(?:option\s*)?([{labels}{labels.lower()}])[.():\s]*", answer)
    if match:
        return match.group(1).upper()
    normalized = normalize(answer)
    options = [normalize(choice) for choice in choices]
    matches = [index for index, choice in enumerate(options) if normalized == choice]
    return chr(ord("A") + matches[0]) if len(matches) == 1 else ""


def make_message(sample: dict, audio_path: Path, think: bool, limit: int) -> list[dict]:
    if think:
        instruction = (
            f"Output the thinking process (less than {limit} words) in <think> </think> "
            "and final answer in <answer> </answer>. Do not output anything after </answer>."
        )
    else:
        instruction = (
            "Directly output the final answer in <answer> </answer>. "
            "Do not output anything after </answer>."
        )
    prompt = (
        f"{sample['question_text']} Please choose the answer from the following options: "
        f"{sample['multi_choice']}. {instruction}"
    )
    return [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": str(audio_path)},
            {"type": "text", "text": prompt},
        ],
    }]


def save(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    samples = read_jsonl(args.data_file)
    results = read_jsonl(args.out_file)
    if [row["id"] for row in results] != [row["id"] for row in samples[: len(results)]]:
        raise ValueError("Existing output is not a prefix of the requested ADQA dataset")

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    for start in tqdm(range(len(results), len(samples), args.batch_size), desc="ADQA"):
        batch = samples[start : start + args.batch_size]
        messages = [
            make_message(row, args.data_root / row["audio_path"], args.think, args.think_max_len)
            for row in batch
        ]
        audios, _, _ = process_mm_info(messages, use_audio_in_video=False)
        texts = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(
            text=texts,
            audio=audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        ).to(model.device).to(torch.bfloat16)
        stopping = StoppingCriteriaList([
            DecodedStringStoppingCriteria(processor.tokenizer, "</answer>", inputs.input_ids.size(1))
        ])
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
                stopping_criteria=stopping,
            )
        outputs = processor.batch_decode(
            generated[:, inputs.input_ids.size(1) :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for sample, output in zip(batch, outputs):
            answer = extract_answer(output)
            choices = sample["multi_choice"]
            gold_text = sample["answer"]
            results.append({
                **sample,
                "gold_text": gold_text,
                "gold_letter": chr(ord("A") + choices.index(gold_text)),
                "answer_prediction": answer,
                "response": answer_letter(answer, choices),
                "model_output_ori": output,
            })
        save(args.out_file, results)

    correct = sum(
        row["response"] == row["gold_letter"]
        or normalize(row["answer_prediction"]) == normalize(row["gold_text"])
        for row in results
    )
    metrics = {"correct": correct, "total": len(results), "accuracy": correct / len(results)}
    args.out_file.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
