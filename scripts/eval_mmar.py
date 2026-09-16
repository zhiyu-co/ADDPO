import argparse
import json
import logging
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
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-Omni on MMAR")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--think-max-len", type=int, default=60)
    return parser.parse_args()


def make_message(sample, audio_root, think, think_max_len):
    choices = f"Please choose the answer from the following options: {sample['choices']}."
    if think:
        instruction = (
            f"Output the thinking process (less than {think_max_len} words) in <think> </think> "
            "and final answer in <answer> </answer>. Do not output anything after </answer>."
        )
    else:
        instruction = (
            "Directly output the final answer in <answer> </answer>. "
            "Do not output anything after </answer>."
        )
    relative_audio = sample["audio_path"].removeprefix("./audio/")
    audio_path = str(Path(audio_root) / relative_audio)
    return [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": f"{sample['question']} {choices} {instruction}"},
        ],
    }]


def extract_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    return (match.group(1) if match else text).strip()


def load_results(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, encoding="utf-8") as reader:
        return [json.loads(line) for line in reader if line.strip()]


def save_results(path, results):
    temporary_path = path + ".partial"
    with open(temporary_path, "w", encoding="utf-8") as writer:
        for result in results:
            writer.write(json.dumps(result, ensure_ascii=False) + "\n")
    os.replace(temporary_path, path)


def main():
    args = parse_args()
    logging.getLogger().setLevel(logging.ERROR)
    with open(args.data_file, encoding="utf-8") as reader:
        samples = json.load(reader)
    results = load_results(args.out_file)
    if [row["id"] for row in results] != [row["id"] for row in samples[:len(results)]]:
        raise ValueError("Existing result is not a prefix of the MMAR metadata")

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    for start in tqdm(range(len(results), len(samples), args.batch_size), desc="MMAR"):
        batch = samples[start:start + args.batch_size]
        messages = [
            make_message(row, args.audio_root, args.think, args.think_max_len)
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
        stopping = StoppingCriteriaList([DecodedStringStoppingCriteria(
            processor.tokenizer, "</answer>", inputs.input_ids.size(1)
        )])
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
                stopping_criteria=stopping,
            )
        generated = generated[:, inputs.input_ids.size(1):]
        outputs = processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for sample, output in zip(batch, outputs):
            result = dict(sample)
            result["answer_prediction"] = extract_answer(output)
            result["model_output_ori"] = output
            results.append(result)
        save_results(args.out_file, results)


if __name__ == "__main__":
    main()
