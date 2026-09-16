import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from tqdm import tqdm
from transformers import HfArgumentParser
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor, StoppingCriteriaList
from qwen_omni_utils import process_mm_info
from utils.generation import DecodedStringStoppingCriteria

@dataclass
class TestArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    model_path: Optional[str] = field(default=None, metadata={"help": "model dir"})
    out_file: Optional[str] = field(default=None, metadata={"help": "output file for test"})
    data_file: Optional[str] = field(default=None, metadata={"help": "test file"})
    force: Optional[bool] = field(default=False, metadata={"help": "force test"})
    batch_size: Optional[int] = field(default=16, metadata={"help": "Batch size for processing"})
    max_new_tokens: Optional[int] = field(default=256, metadata={"help": "Maximum generated completion length"})
    think: Optional[bool] = field(default=False, metadata={"help": "whether think step by step"})
    think_max_len: Optional[int] = field(
        default=50, metadata={"help": "Max length of think process, 0 for unlimit"}
    )

    def __post_init__(self):
        if self.model_path is None:
            raise ValueError("model_path is required")
        if self.data_file is None:
            raise ValueError("data_file is required")
        if self.out_file is None:
            raise ValueError("out_file is required")


def _get_message(obj_dict, think=False, think_max_len=50):
    choice_str = f"Please choose the answer from the following options: {obj_dict['choices']}."
    question = obj_dict["question"].replace("video", "audio")
    if think:
        if think_max_len <= 0:
            question_template = f"{question} {choice_str} Output the thinking process in <think> </think> and final answer in <answer> </answer>. Do not output anything after </answer>."
        else:
            # Keep this byte-for-byte aligned with src/dataset/dataset.py,
            # which constructs the prompts used during GRPO training.
            question_template = f"{question} {choice_str} Output the thinking process (less than {think_max_len} words) in <think> </think> and final answer in <answer> </answer>. Do not output anything after </answer>."
    else:
        question_template = f"{question} {choice_str} Directly output the final answer in <answer> </answer>. Do not output anything after </answer>."
    audio_path = obj_dict.get("audio", obj_dict["audio_id"])
    message = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": question_template},
            ],
        }
    ]
    return message


def main():
    parser = HfArgumentParser(TestArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    transformers.logging.set_verbosity_info()
    logging.info(data_args)

    with open(data_args.data_file, "r", encoding="utf8") as reader:
        datas = json.load(reader)

    final_output = []
    if not data_args.force and os.path.exists(data_args.out_file) and os.path.getsize(data_args.out_file) > 0:
        with open(data_args.out_file, "r", encoding="utf8") as reader:
            final_output = json.load(reader)
        if len(final_output) == len(datas):
            logging.info(f"The completed {data_args.out_file} exists. Do not regenerate it.")
            return
        expected_ids = [row.get("id") for row in datas[: len(final_output)]]
        result_ids = [row.get("id") for row in final_output]
        if result_ids != expected_ids:
            raise ValueError("Existing partial result is not a prefix of the requested dataset")
        logging.info(f"Resuming {data_args.out_file} at sample {len(final_output)}/{len(datas)}")

    qwen2_omni_processor = Qwen2_5OmniProcessor.from_pretrained(data_args.model_path)
    qwen2_omni_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        data_args.model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa"
    )
    qwen2_omni_model.eval()

    def extract_answer(output_str):
        answer_pattern = r"<answer>(.*?)</answer>"
        match = re.search(answer_pattern, output_str)
        if match:
            return match.group(1)
        return output_str

    batch_size = data_args.batch_size
    for i in tqdm(range(len(final_output), len(datas), batch_size)):
        batch_data = datas[i : i + batch_size]

        batch_messages = []
        batch_audios = []
        for bd in batch_data:
            batch_messages.append(_get_message(bd, data_args.think, data_args.think_max_len))

        batch_audios, _, _ = process_mm_info(batch_messages, use_audio_in_video=False)

        text = qwen2_omni_processor.apply_chat_template(batch_messages, add_generation_prompt=True, tokenize=False)
        inputs = qwen2_omni_processor(
            text=text, audio=batch_audios, sampling_rate=16000, return_tensors="pt", padding=True
        ).to(qwen2_omni_model.device).to(torch.bfloat16)

        answer_stopping_criteria = StoppingCriteriaList(
            [
                DecodedStringStoppingCriteria(
                    qwen2_omni_processor.tokenizer,
                    "</answer>",
                    inputs.input_ids.size(1),
                )
            ]
        )
        generated_ids = qwen2_omni_model.generate(
            **inputs,
            max_new_tokens=data_args.max_new_tokens,
            eos_token_id=qwen2_omni_processor.tokenizer.eos_token_id,
            pad_token_id=qwen2_omni_processor.tokenizer.pad_token_id,
            stopping_criteria=answer_stopping_criteria,
        )
        generated_ids = generated_ids[:, inputs.input_ids.size(1) :]
        response = qwen2_omni_processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for input_example, model_output in zip(batch_data, response):
            result = dict(input_example)
            result["model_output"] = extract_answer(model_output).strip()
            result["model_output_ori"] = model_output
            final_output.append(result)

        output_path = data_args.out_file
        temporary_path = output_path + ".partial"
        with open(temporary_path, "w", encoding="utf8") as writer:
            json.dump(final_output, writer, ensure_ascii=False, indent=2)
        os.replace(temporary_path, output_path)
        print(f"Processed batch {i//batch_size + 1}/{(len(datas) + batch_size - 1)//batch_size}")

    print(f"Results saved to {data_args.out_file}")


if __name__ == "__main__":
    main()
