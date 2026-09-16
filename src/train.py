"""Train Qwen2.5-Omni with GRPO or AD²PO."""

import logging
import os
from dataclasses import dataclass, field
from typing import Literal, Optional

import transformers
from transformers import HfArgumentParser
from trl import GRPOConfig

from dataset.dataset import AudioDataset
from trainer.grpo_trainer import GRPOTrainer
from utils.rewards import accuracy_reward, format_reward


@dataclass
class TrainingArguments:
    model_name_or_path: str = field(metadata={"help": "Qwen2.5-Omni model or checkpoint"})
    data_file: str = field(metadata={"help": "Prepared JSONL training set"})
    output_dir: str = field(metadata={"help": "Directory for checkpoints and logs"})
    deepspeed: Optional[str] = field(default=None, metadata={"help": "DeepSpeed JSON config"})
    method: Literal["grpo", "trajectory", "token", "joint"] = field(
        default="joint",
        metadata={"help": "GRPO baseline or an AD²PO weighting variant"},
    )
    think_max_len: int = field(default=128)
    beta: float = field(default=0.01, metadata={"help": "Reference-model KL coefficient"})
    num_generations: int = field(default=8)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=8)
    max_steps: int = field(default=338)
    save_steps: int = field(default=338)
    logging_steps: int = field(default=1)
    learning_rate: float = field(default=1e-6)
    seed: int = field(default=42)
    freeze_audio_encoder: bool = field(default=True)
    report_to: Literal["swanlab", "none"] = field(default="swanlab")
    run_name: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.num_generations < 2:
            raise ValueError("num_generations must be at least 2")
        if self.method not in {"grpo", "trajectory", "token", "joint"}:
            raise ValueError(f"Unsupported method: {self.method}")


def main() -> None:
    args = HfArgumentParser(TrainingArguments).parse_args_into_dataclasses()[0]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    transformers.logging.set_verbosity_info()
    logging.info("Training configuration: %s", args)

    use_trajectory_weight = args.method in {"trajectory", "joint"}
    use_token_weight = args.method in {"token", "joint"}
    run_name = args.run_name or f"qwen2.5-omni-7b-{args.method}-seed{args.seed}"

    config = GRPOConfig(
        output_dir=args.output_dir,
        run_name=run_name,
        deepspeed=args.deepspeed,
        max_steps=args.max_steps,
        save_strategy="steps" if args.save_steps > 0 else "no",
        save_steps=max(args.save_steps, 1),
        save_only_model=True,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_prompt_length=2048,
        max_completion_length=128,
        temperature=1.0,
        beta=args.beta,
        reward_weights=[2.0, 1.0],
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
        report_to="swanlab" if args.report_to == "swanlab" else "none",
    )

    swanlab_run = None
    if args.report_to == "swanlab" and int(os.getenv("RANK", "0")) == 0:
        import swanlab

        swanlab_run = swanlab.init(
            project=os.getenv("SWANLAB_PROJECT_NAME", "AD2PO"),
            name=run_name,
            log_dir=os.getenv("SWANLAB_LOG_DIR"),
            mode=os.getenv("SWANLAB_MODE", "online"),
            config={
                **vars(args),
                "trajectory_weighting": use_trajectory_weight,
                "token_weighting": use_token_weight,
            },
        )

    dataset = AudioDataset(args.data_file, is_think=True, think_max_len=args.think_max_len)
    trainer = GRPOTrainer(
        model=args.model_name_or_path,
        reward_funcs=[accuracy_reward, format_reward],
        args=config,
        train_dataset=dataset,
        think=True,
        freeze_audio_encoder=args.freeze_audio_encoder,
        ad2po_trajectory_weighting=use_trajectory_weight,
        ad2po_token_weighting=use_token_weight,
    )

    try:
        trainer.train()
        trainer.save_model(args.output_dir)
    finally:
        if swanlab_run is not None:
            swanlab.finish()


if __name__ == "__main__":
    main()
