# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by zhaoshuaijiang8@gmail.com from KE-Team

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union

import numpy as np
import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniProcessor,
    Qwen2AudioForConditionalGeneration,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    StoppingCriteriaList,
    Trainer,
    TrainerCallback,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
import trl.models.utils as trl_model_utils
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, selective_log_softmax
from trl.trainer.callbacks import SyncRefModelCallback
from qwen_omni_utils import process_mm_info
from utils.generation import DecodedStringStoppingCriteria
if is_peft_available():
    from peft import PeftConfig, get_peft_model


def _add_deepspeed_hooks_compat(model):
    """Restore ZeRO-3 hooks across old TRL and current DeepSpeed versions."""
    if not hasattr(model, "optimizer") or model.optimizer is None:
        return
    if hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    else:
        optimizer_offload = model.optimizer

    register_hooks = getattr(optimizer_offload, "_register_hooks_recursively", None)
    if register_hooks is None:
        register_hooks = optimizer_offload._register_deepspeed_module
    register_hooks(optimizer_offload.module)


# TRL 0.15 calls a hook-registration method removed by newer DeepSpeed. The
# generation context resolves add_hooks at runtime, so keep the shim local to
# this process instead of modifying the installed dependency.
trl_model_utils.add_hooks = _add_deepspeed_hooks_compat

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


def randomly_silence_audio(audio: Union[np.ndarray, torch.Tensor], silence_ratio: float):
    """Randomly zero waveform samples while preserving shape and dtype."""
    if not 0.0 <= silence_ratio <= 1.0:
        raise ValueError("audio dependency silence ratio must be in [0, 1]")
    if isinstance(audio, torch.Tensor):
        silent_mask = torch.rand(audio.shape, device=audio.device) < silence_ratio
        return audio.masked_fill(silent_mask, 0)
    silent_mask = np.random.random_sample(audio.shape) < silence_ratio
    return np.where(silent_mask, np.zeros((), dtype=audio.dtype), audio)


def group_top_fraction_token_mask(
    token_dependency: torch.Tensor,
    token_mask: torch.Tensor,
    num_generations: int,
    fraction: float,
) -> torch.Tensor:
    """Select exactly the highest-dependency fraction pooled across each generation group."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("top token fraction must be in (0, 1]")
    if token_dependency.shape != token_mask.shape:
        raise ValueError("token dependency and mask must have identical shapes")
    if token_dependency.size(0) % num_generations:
        raise ValueError("trajectory count must be divisible by num_generations")
    sequence_length = token_dependency.size(1)
    grouped_dependency = token_dependency.detach().reshape(-1, num_generations * sequence_length)
    grouped_valid = token_mask.bool().reshape(-1, num_generations * sequence_length)
    grouped_selected = torch.zeros_like(grouped_valid)
    for group_index in range(grouped_dependency.size(0)):
        valid_indices = grouped_valid[group_index].nonzero(as_tuple=False).squeeze(1)
        if valid_indices.numel() == 0:
            continue
        keep_count = max(1, int(np.ceil(valid_indices.numel() * fraction)))
        valid_scores = grouped_dependency[group_index, valid_indices]
        top_indices = valid_scores.topk(keep_count, largest=True, sorted=False).indices
        grouped_selected[group_index, valid_indices[top_indices]] = True
    return grouped_selected.reshape_as(token_mask)


def trajectory_top_fraction_token_mask(
    token_dependency: torch.Tensor,
    token_mask: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    """Select the highest-dependency fraction independently within every trajectory."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("top token fraction must be in (0, 1]")
    if token_dependency.shape != token_mask.shape:
        raise ValueError("token dependency and mask must have identical shapes")
    valid = token_mask.bool()
    selected = torch.zeros_like(valid)
    for trajectory_index in range(token_dependency.size(0)):
        valid_indices = valid[trajectory_index].nonzero(as_tuple=False).squeeze(1)
        if valid_indices.numel() == 0:
            continue
        keep_count = max(1, int(np.ceil(valid_indices.numel() * fraction)))
        valid_scores = token_dependency[trajectory_index, valid_indices].detach()
        top_indices = valid_scores.topk(keep_count, largest=True, sorted=False).indices
        selected[trajectory_index, valid_indices[top_indices]] = True
    return selected


def binary_rank_within_trajectory_token_weights(
    token_dependency: torch.Tensor,
    token_mask: torch.Tensor,
    low_weight: float = 0.7,
    high_weight: float = 1.3,
) -> torch.Tensor:
    """Assign fixed weights to the lower and upper halves in each trajectory.

    For odd sequence lengths the single median token receives weight 1.0. This
    makes the valid-token weights sum exactly to the trajectory length when
    low_weight + high_weight == 2. Padding receives zero weight.
    """
    if token_dependency.shape != token_mask.shape:
        raise ValueError("token dependency and mask must have identical shapes")
    if not 0.0 <= low_weight <= 1.0 <= high_weight:
        raise ValueError("binary token weights must satisfy 0 <= low <= 1 <= high")
    if abs((low_weight + high_weight) - 2.0) > 1e-6:
        raise ValueError("binary token weights must sum to 2 to preserve sequence weight")

    valid = token_mask.bool()
    weights = valid.to(dtype=torch.float32)
    for trajectory_index in range(token_dependency.size(0)):
        valid_indices = valid[trajectory_index].nonzero(as_tuple=False).squeeze(1)
        half_count = valid_indices.numel() // 2
        if half_count == 0:
            continue
        valid_scores = token_dependency[trajectory_index, valid_indices].detach()
        ranked = valid_scores.argsort()
        lower_indices = valid_indices[ranked[:half_count]]
        upper_indices = valid_indices[ranked[-half_count:]]
        weights[trajectory_index, lower_indices] = low_weight
        weights[trajectory_index, upper_indices] = high_weight
    return weights.masked_fill(~valid, 0.0)


def k3_audio_dependency(
    audio_logps: torch.Tensor,
    silent_logps: torch.Tensor,
    token_mask: torch.Tensor,
    token_kl_max: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return trajectory-level and token-level KL_k3(audio || silence).

    Completions are sampled from the audio-conditioned policy.  Evaluating the
    same completion under a silent input makes ``audio_logps`` log p(x|a) and
    ``silent_logps`` log q(x|silence), so k3 is exp(log q-log p)-(log q-log p)-1.
    """
    dependency_dtype = torch.float32 if token_kl_max > 0 else torch.float64
    log_ratio = silent_logps.to(dependency_dtype) - audio_logps.detach().to(dependency_dtype)
    token_k3 = torch.exp(log_ratio) - log_ratio - 1.0
    # A non-positive threshold disables clipping. This is useful for the
    # trajectory-reward variant, where the raw token K3 values are averaged
    # before the eight sampled trajectories are normalized together.
    if token_kl_max > 0:
        token_k3 = token_k3.clamp(max=token_kl_max)
    denominator = token_mask.sum(dim=1).clamp_min(1)
    trajectory_k3 = (token_k3 * token_mask).sum(dim=1) / denominator
    return trajectory_k3, token_k3


def redistribute_advantages_by_audio_dependency(
    advantages: torch.Tensor,
    token_dependency: torch.Tensor,
    token_mask: torch.Tensor,
    token_kl_max: float,
    mix: float = 1.0,
    weight_min: float = 0.3,
    weight_max: float = 1.7,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Redistribute trajectory advantage and directly clamp token weights.

    ``mix=0`` recovers ordinary GRPO and ``mix=1`` uses pure audio-dependency
    weights.  A trajectory whose dependency scores are all zero falls back to
    uniform weights so its policy gradient is not discarded.
    """
    if not 0.0 <= mix <= 1.0:
        raise ValueError("audio dependency advantage mix must be in [0, 1]")
    if not 0.0 <= weight_min <= 1.0 <= weight_max:
        raise ValueError("audio dependency advantage weight bounds must satisfy 0 <= min <= 1 <= max")
    mask = token_mask.to(token_dependency.dtype)
    score = token_dependency.detach().clamp(min=0.0, max=token_kl_max) / token_kl_max
    raw_weights = ((1.0 - mix) + mix * score) * mask
    token_count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean_weight = raw_weights.sum(dim=1, keepdim=True) / token_count
    normalized_weights = torch.where(
        mean_weight > eps,
        raw_weights / mean_weight.clamp_min(eps),
        mask,
    )
    # Intentionally do not renormalize after the clamp. Sparse/concentrated
    # trajectories therefore receive a smaller total policy-gradient update.
    normalized_weights = normalized_weights.clamp(min=weight_min, max=weight_max) * mask
    return advantages.unsqueeze(1) * normalized_weights, normalized_weights


def correctness_gated_audio_reward(
    dependency: torch.Tensor,
    correctness: torch.Tensor,
    num_generations: int,
    reward_lambda: float,
    reward_mu: float,
    eps: float = 1e-4,
    min_group_range: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply within-prompt min-max normalization and correctness gating."""
    grouped = dependency.view(-1, num_generations)
    group_min = grouped.min(dim=1, keepdim=True).values
    group_range = grouped.max(dim=1, keepdim=True).values - group_min
    normalized = (grouped - group_min) / (group_range + eps)
    normalized = torch.where(group_range >= min_group_range, normalized, torch.zeros_like(normalized)).reshape(-1)
    normalized = normalized.float()
    gated = reward_lambda * correctness * normalized - reward_mu * (1.0 - correctness) * normalized
    return gated, normalized


def full_vocab_forward_kl(
    audio_logits: torch.Tensor,
    silent_logits: torch.Tensor,
    chunk_size: int = 4,
) -> torch.Tensor:
    """Compute token-wise KL(P_audio || P_silence) over the full vocabulary.

    Sequence chunks bound the float32 softmax working set. Both inputs are
    detached because audio dependency is a training signal, not a differentiable
    auxiliary loss.
    """
    if audio_logits.shape != silent_logits.shape:
        raise ValueError("Audio and silence completion logits must have identical shapes")
    chunks = []
    for start in range(0, audio_logits.size(1), chunk_size):
        end = min(start + chunk_size, audio_logits.size(1))
        audio_log_probs = torch.log_softmax(audio_logits[:, start:end].detach().float(), dim=-1)
        silent_log_probs = torch.log_softmax(silent_logits[:, start:end].detach().float(), dim=-1)
        token_kl = (audio_log_probs.exp() * (audio_log_probs - silent_log_probs)).sum(dim=-1)
        chunks.append(token_kl.clamp_min(0.0))
    return torch.cat(chunks, dim=1)


def full_vocab_jensen_shannon_divergence(
    audio_logits: torch.Tensor,
    silent_logits: torch.Tensor,
    chunk_size: int = 4,
) -> torch.Tensor:
    """Compute token-wise Jensen-Shannon divergence over the full vocabulary.

    JSD(P, Q) = 0.5 KL(P || M) + 0.5 KL(Q || M), where M = (P + Q) / 2.
    Unlike directional KL, this score is symmetric and bounded by log(2).
    Inputs are detached because it is used only to allocate policy gradients.
    """
    if audio_logits.shape != silent_logits.shape:
        raise ValueError("Audio and silence completion logits must have identical shapes")
    chunks = []
    log_two = 0.6931471805599453
    for start in range(0, audio_logits.size(1), chunk_size):
        end = min(start + chunk_size, audio_logits.size(1))
        audio_log_probs = torch.log_softmax(audio_logits[:, start:end].detach().float(), dim=-1)
        silent_log_probs = torch.log_softmax(silent_logits[:, start:end].detach().float(), dim=-1)
        mixture_log_probs = torch.logaddexp(audio_log_probs, silent_log_probs) - log_two
        audio_kl = (audio_log_probs.exp() * (audio_log_probs - mixture_log_probs)).sum(dim=-1)
        silent_kl = (silent_log_probs.exp() * (silent_log_probs - mixture_log_probs)).sum(dim=-1)
        # JSD is non-negative analytically; remove only float32 round-off below zero.
        chunks.append((0.5 * (audio_kl + silent_kl)).clamp_min(0.0))
    return torch.cat(chunks, dim=1)


def full_vocab_absolute_entropy_delta(
    audio_logits: torch.Tensor,
    silent_logits: torch.Tensor,
    chunk_size: int = 4,
) -> torch.Tensor:
    """Compute token-wise |H(P_audio) - H(P_silence)| over the full vocabulary."""
    if audio_logits.shape != silent_logits.shape:
        raise ValueError("Audio and silence completion logits must have identical shapes")
    chunks = []
    for start in range(0, audio_logits.size(1), chunk_size):
        end = min(start + chunk_size, audio_logits.size(1))
        audio_log_probs = torch.log_softmax(audio_logits[:, start:end].detach().float(), dim=-1)
        silent_log_probs = torch.log_softmax(silent_logits[:, start:end].detach().float(), dim=-1)
        audio_entropy = -(audio_log_probs.exp() * audio_log_probs).sum(dim=-1)
        silent_entropy = -(silent_log_probs.exp() * silent_log_probs).sum(dim=-1)
        chunks.append((audio_entropy - silent_entropy).abs())
    return torch.cat(chunks, dim=1)


def full_vocab_token_entropy(
    logits: torch.Tensor,
    chunk_size: int = 4,
) -> torch.Tensor:
    """Compute full-vocabulary entropy at each completion-token position."""
    chunks = []
    for start in range(0, logits.size(1), chunk_size):
        end = min(start + chunk_size, logits.size(1))
        log_probs = torch.log_softmax(logits[:, start:end].detach().float(), dim=-1)
        chunks.append(-(log_probs.exp() * log_probs).sum(dim=-1))
    return torch.cat(chunks, dim=1)


def minmax_trajectory_advantage_weights(
    dependency: torch.Tensor,
    num_generations: int,
    weight_min: float = 0.6,
    weight_max: float = 1.4,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map within-prompt trajectory dependency to a bounded advantage weight."""
    grouped = dependency.view(-1, num_generations)
    group_min = grouped.min(dim=1, keepdim=True).values
    group_range = grouped.max(dim=1, keepdim=True).values - group_min
    normalized = ((grouped - group_min) / (group_range + eps)).float()
    weights = weight_min + (weight_max - weight_min) * normalized
    return weights.reshape(-1), normalized.reshape(-1)


def minmax_centered_trajectory_advantage_weights(
    dependency: torch.Tensor,
    num_generations: int,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Min-max and center trajectory scores within each generation group.

    For each prompt's K trajectories, z_k = minmax(D_k) and
    w_k = 1 + z_k - mean_j(z_j). This keeps every group's mean trajectory
    weight exactly one and bounds all weights to [0, 2]. A constant-score
    group carries no ranking information and falls back to uniform weights.
    """
    grouped = dependency.detach().float().view(-1, num_generations)
    group_min = grouped.min(dim=1, keepdim=True).values
    group_range = grouped.max(dim=1, keepdim=True).values - group_min
    normalized = (grouped - group_min) / group_range.clamp_min(eps)
    normalized = torch.where(
        group_range > eps, normalized, torch.zeros_like(normalized)
    )
    weights = 1.0 + normalized - normalized.mean(dim=1, keepdim=True)
    return weights.reshape(-1), normalized.reshape(-1)


def minmax_within_trajectory_token_weights(
    token_dependency: torch.Tensor,
    token_mask: torch.Tensor,
    weight_min: float = 0.7,
    weight_max: float = 1.3,
    eps: float = 1e-4,
    min_range: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Min-max audio dependency over valid tokens in each trajectory.

    Degenerate trajectories fall back to weight 1 instead of suppressing the
    whole trajectory. Padding receives zero weight and is excluded from stats.
    """
    if token_dependency.shape != token_mask.shape:
        raise ValueError("token dependency and mask must have identical shapes")
    valid = token_mask.bool()
    masked_min = token_dependency.detach().masked_fill(~valid, torch.inf).min(dim=1, keepdim=True).values
    masked_max = token_dependency.detach().masked_fill(~valid, -torch.inf).max(dim=1, keepdim=True).values
    has_tokens = valid.any(dim=1, keepdim=True)
    score_range = masked_max - masked_min
    normalized = (token_dependency.detach() - masked_min) / (score_range + eps)
    normalized = torch.where(has_tokens & (score_range >= min_range), normalized, torch.full_like(normalized, 0.5))
    normalized = normalized.masked_fill(~valid, 0.0).float()
    weights = weight_min + (weight_max - weight_min) * normalized
    weights = torch.where(valid, weights, torch.zeros_like(weights))
    return weights, normalized


def audio_entropy_within_trajectory_token_weights(
    token_dependency: torch.Tensor,
    token_entropy: torch.Tensor,
    token_mask: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build unclipped, mean-preserving token weights from KL and entropy.

    Audio dependency and policy entropy are independently min-max normalized
    within each trajectory, then averaged with equal weight. The combined
    score is divided by its valid-token mean so each nonempty trajectory keeps
    the original total advantage scale. No lower or upper clipping is applied.
    """
    if token_dependency.shape != token_entropy.shape or token_dependency.shape != token_mask.shape:
        raise ValueError("token dependency, entropy, and mask must have identical shapes")

    def normalize(values: torch.Tensor) -> torch.Tensor:
        valid = token_mask.bool()
        detached = values.detach()
        row_min = detached.masked_fill(~valid, torch.inf).min(dim=1, keepdim=True).values
        row_max = detached.masked_fill(~valid, -torch.inf).max(dim=1, keepdim=True).values
        row_range = row_max - row_min
        has_tokens = valid.any(dim=1, keepdim=True)
        normalized = (detached - row_min) / (row_range + eps)
        # Constant-score trajectories carry no ranking information; assigning
        # 0.5 makes the subsequent mean normalization recover uniform weight 1.
        normalized = torch.where(
            has_tokens & (row_range >= eps), normalized, torch.full_like(normalized, 0.5)
        )
        return normalized.masked_fill(~valid, 0.0).float()

    dependency_normalized = normalize(token_dependency)
    entropy_normalized = normalize(token_entropy)
    combined_score = 0.5 * dependency_normalized + 0.5 * entropy_normalized
    mask = token_mask.to(combined_score.dtype)
    token_count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    score_mean = (combined_score * mask).sum(dim=1, keepdim=True) / token_count
    weights = torch.where(score_mean > eps, combined_score / score_mean.clamp_min(eps), mask) * mask
    return weights, combined_score, dependency_normalized, entropy_normalized


def minmax_centered_token_weights(
    token_scores: torch.Tensor,
    token_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Min-max and center token scores within each sequence.

    For valid tokens in a trajectory, first compute
    z_t = (s_t - min_j(s_j)) / (max_j(s_j) - min_j(s_j)), then use
    w_t = 1 + z_t - mean_j(z_j). Thus the valid-token mean is exactly one,
    the trajectory's total token weight is preserved, and weights stay in
    [0, 2]. A constant-score trajectory falls back to uniform weight one.
    """
    if token_scores.shape != token_mask.shape:
        raise ValueError("token scores and mask must have identical shapes")
    valid = token_mask.bool()
    mask = valid.to(dtype=torch.float32)
    scores = token_scores.detach().float().masked_fill(~valid, 0.0)
    has_tokens = valid.any(dim=1, keepdim=True)
    row_min = scores.masked_fill(~valid, torch.inf).min(dim=1, keepdim=True).values
    row_max = scores.masked_fill(~valid, -torch.inf).max(dim=1, keepdim=True).values
    row_min = torch.where(has_tokens, row_min, torch.zeros_like(row_min))
    row_max = torch.where(has_tokens, row_max, torch.zeros_like(row_max))
    row_range = row_max - row_min

    normalized = (scores - row_min) / row_range.clamp_min(eps)
    normalized = torch.where(
        has_tokens & (row_range > eps), normalized, torch.zeros_like(normalized)
    ).masked_fill(~valid, 0.0)
    token_count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    normalized_mean = (normalized * mask).sum(dim=1, keepdim=True) / token_count
    weights = (1.0 + normalized - normalized_mean) * mask
    return weights.masked_fill(~valid, 0.0)


def completion_mask_through_stop(
    completion_ids: torch.Tensor,
    eos_token_id: int,
    pad_token_id: int,
    stop_token_ids: list[int],
) -> torch.Tensor:
    """Mask each completion through its first EOS or complete stop sequence."""
    batch_size, sequence_length = completion_ids.shape
    device = completion_ids.device
    end_idx = torch.full((batch_size,), sequence_length, dtype=torch.long, device=device)

    is_eos = completion_ids.eq(eos_token_id)
    has_eos = is_eos.any(dim=1)
    end_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]

    # Per-row stopping criteria mark a finished row as done while other rows
    # continue. Generate fills the remainder of that row with pad tokens.
    is_pad = completion_ids.eq(pad_token_id)
    has_pad = is_pad.any(dim=1)
    first_pad_end = is_pad.int().argmax(dim=1) - 1
    end_idx[has_pad] = torch.minimum(end_idx[has_pad], first_pad_end[has_pad])

    stop_length = len(stop_token_ids)
    if stop_length and sequence_length >= stop_length:
        stop_pattern = torch.tensor(stop_token_ids, dtype=completion_ids.dtype, device=device)
        matches = completion_ids.unfold(1, stop_length, 1).eq(stop_pattern).all(dim=-1)
        has_stop = matches.any(dim=1)
        first_stop_end = matches.int().argmax(dim=1) + stop_length - 1
        end_idx[has_stop] = torch.minimum(end_idx[has_stop], first_stop_end[has_stop])

    positions = torch.arange(sequence_length, device=device).expand(batch_size, -1)
    return positions.le(end_idx.unsqueeze(1)).int()


# Based on R1-V code base, https://github.com/Deep-Agent/R1-V/blob/main/src/r1-v/src/open_r1/trainer/grpo_trainer.py
class GRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        think: Optional[bool] = False,
        freeze_audio_encoder: bool = False,
        ad2po_trajectory_weighting: bool = False,
        ad2po_token_weighting: bool = False,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs.setdefault("torch_dtype", torch.bfloat16)
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            if "Qwen2-Audio" in model_id:
                model = Qwen2AudioForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-Omni" in model_id:
                model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model, low_cpu_mem_usage=True, attn_implementation="sdpa", **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if freeze_audio_encoder:
            audio_tower = getattr(model, "audio_tower", None)
            if audio_tower is None:
                raise ValueError("freeze_audio_encoder=True but the model has no audio_tower")
            audio_tower.requires_grad_(False)

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Keep the original identifier independently of PEFT/DeepSpeed wrappers;
        # those wrappers do not consistently forward `name_or_path`.
        self.model_id = model_id

        # Reference model, when need KL
        self.ref_model = None
        deferred_ref_model_id = None
        if args.beta > 0:
            if is_deepspeed_zero3_enabled():
                # Loading the reference model before DeepSpeed initializes the
                # trainable model's optimizer makes both 7B models overlap with
                # Adam's temporary state allocations. Defer it to keep the
                # initialization peak below the container memory limit.
                deferred_ref_model_id = model_id
            elif not is_peft_model(model):
                # If PEFT configuration is not provided, create a reference model based on the initial model.
                self.ref_model = create_reference_model(model)
            else:
                # If PEFT is used, the reference model is not needed since the adapter can be disabled
                # to revert to the initial model.
                self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen2-Audio" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                processing_class.pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            elif "Qwen2.5-Omni" in model_id:
                processing_class = Qwen2_5OmniProcessor.from_pretrained(model_id, padding_side='left')
                processing_class.pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.beta = args.beta
        self.think = think
        # The public release exposes only the paper's bounded JSD variants.
        # Legacy experiment switches remain internal below until the trainer is
        # rebased on a newer TRL implementation.
        audio_dependency_reward = False
        audio_dependency_lambda = 1.0
        audio_dependency_mu = 1.0
        audio_dependency_token_kl_max = 0.0
        audio_dependency_token_advantage = False
        audio_dependency_advantage_mix = 1.0
        audio_dependency_advantage_weight_min = 0.0
        audio_dependency_advantage_weight_max = 2.0
        audio_dependency_full_vocab_trajectory_advantage = False
        audio_dependency_full_vocab_token_weight_advantage = False
        audio_dependency_full_vocab_entropy_token_advantage = False
        audio_dependency_full_vocab_js_token_advantage = ad2po_token_weighting
        audio_dependency_full_vocab_js_trajectory_advantage = ad2po_trajectory_weighting
        audio_dependency_full_vocab_binary_token_advantage = False
        audio_dependency_trajectory_token_weight_min = 0.0
        audio_dependency_trajectory_token_weight_max = 2.0
        audio_dependency_entropy_delta_trajectory_advantage = False
        audio_dependency_trajectory_weight_min = 0.0
        audio_dependency_trajectory_weight_max = 2.0
        reasoning_entropy_trajectory_advantage = False
        reasoning_entropy_trajectory_weight_min = 0.0
        reasoning_entropy_trajectory_weight_max = 2.0
        audio_dependency_silence_ratio = 1.0
        audio_dependency_full_vocab_group_top_token_advantage = False
        audio_dependency_full_vocab_trajectory_top_token_advantage = False
        audio_dependency_top_token_fraction = 1.0
        correctness_uncertainty_advantage = False
        self.audio_dependency_reward = audio_dependency_reward
        self.audio_dependency_lambda = audio_dependency_lambda
        self.audio_dependency_mu = audio_dependency_mu
        self.audio_dependency_token_kl_max = audio_dependency_token_kl_max
        self.audio_dependency_token_advantage = audio_dependency_token_advantage
        self.audio_dependency_advantage_mix = audio_dependency_advantage_mix
        self.audio_dependency_advantage_weight_min = audio_dependency_advantage_weight_min
        self.audio_dependency_advantage_weight_max = audio_dependency_advantage_weight_max
        self.audio_dependency_full_vocab_trajectory_advantage = audio_dependency_full_vocab_trajectory_advantage
        self.audio_dependency_full_vocab_token_weight_advantage = audio_dependency_full_vocab_token_weight_advantage
        self.audio_dependency_full_vocab_entropy_token_advantage = audio_dependency_full_vocab_entropy_token_advantage
        self.audio_dependency_full_vocab_js_token_advantage = audio_dependency_full_vocab_js_token_advantage
        self.audio_dependency_full_vocab_js_trajectory_advantage = audio_dependency_full_vocab_js_trajectory_advantage
        self.audio_dependency_full_vocab_binary_token_advantage = audio_dependency_full_vocab_binary_token_advantage
        self.audio_dependency_trajectory_token_weight_min = audio_dependency_trajectory_token_weight_min
        self.audio_dependency_trajectory_token_weight_max = audio_dependency_trajectory_token_weight_max
        self.audio_dependency_entropy_delta_trajectory_advantage = audio_dependency_entropy_delta_trajectory_advantage
        self.audio_dependency_trajectory_weight_min = audio_dependency_trajectory_weight_min
        self.audio_dependency_trajectory_weight_max = audio_dependency_trajectory_weight_max
        self.reasoning_entropy_trajectory_advantage = reasoning_entropy_trajectory_advantage
        self.reasoning_entropy_trajectory_weight_min = reasoning_entropy_trajectory_weight_min
        self.reasoning_entropy_trajectory_weight_max = reasoning_entropy_trajectory_weight_max
        self.audio_dependency_silence_ratio = audio_dependency_silence_ratio
        self.audio_dependency_full_vocab_group_top_token_advantage = audio_dependency_full_vocab_group_top_token_advantage
        self.audio_dependency_full_vocab_trajectory_top_token_advantage = audio_dependency_full_vocab_trajectory_top_token_advantage
        self.audio_dependency_top_token_fraction = audio_dependency_top_token_fraction
        self.correctness_uncertainty_advantage = correctness_uncertainty_advantage
        self.audio_dependency_full_vocab_enabled = (
            self.audio_dependency_full_vocab_trajectory_advantage
            or self.audio_dependency_full_vocab_token_weight_advantage
            or self.audio_dependency_full_vocab_entropy_token_advantage
            or self.audio_dependency_full_vocab_js_token_advantage
            or self.audio_dependency_full_vocab_js_trajectory_advantage
            or self.audio_dependency_full_vocab_binary_token_advantage
            or self.audio_dependency_entropy_delta_trajectory_advantage
            or self.audio_dependency_full_vocab_group_top_token_advantage
            or self.audio_dependency_full_vocab_trajectory_top_token_advantage
        )
        if self.reasoning_entropy_trajectory_advantage and not self.audio_dependency_full_vocab_enabled:
            raise ValueError(
                "Reasoning-entropy trajectory weighting currently requires a full-vocabulary audio mode"
            )
        self.audio_dependency_enabled = (
            self.audio_dependency_reward
            or self.audio_dependency_token_advantage
            or self.audio_dependency_full_vocab_enabled
        )
        enabled_legacy_audio_objectives = sum((
            self.audio_dependency_reward,
            self.audio_dependency_token_advantage,
        ))
        enabled_full_vocab_objectives = sum((
            self.audio_dependency_full_vocab_trajectory_advantage,
            self.audio_dependency_full_vocab_token_weight_advantage,
            self.audio_dependency_full_vocab_entropy_token_advantage,
            self.audio_dependency_full_vocab_js_token_advantage,
            self.audio_dependency_full_vocab_js_trajectory_advantage,
            self.audio_dependency_full_vocab_binary_token_advantage,
            self.audio_dependency_entropy_delta_trajectory_advantage,
            self.audio_dependency_full_vocab_group_top_token_advantage,
            self.audio_dependency_full_vocab_trajectory_top_token_advantage,
        ))
        if enabled_legacy_audio_objectives > 1 or (
            enabled_legacy_audio_objectives and enabled_full_vocab_objectives
        ):
            raise ValueError(
                "Legacy audio dependency reward/token advantage cannot be combined with other audio dependency modes"
            )
        if (
            self.audio_dependency_full_vocab_trajectory_advantage
            and self.audio_dependency_entropy_delta_trajectory_advantage
        ):
            raise ValueError("KL and entropy-delta trajectory scores are mutually exclusive")
        if (
            self.audio_dependency_full_vocab_trajectory_advantage
            and self.audio_dependency_full_vocab_js_trajectory_advantage
        ):
            raise ValueError("KL and Jensen-Shannon trajectory scores are mutually exclusive")
        if self.audio_dependency_token_advantage and self.audio_dependency_token_kl_max <= 0:
            raise ValueError("Token-level audio advantage requires a positive audio_dependency_token_kl_max")
        if not 0.0 <= self.audio_dependency_advantage_mix <= 1.0:
            raise ValueError("audio_dependency_advantage_mix must be in [0, 1]")
        if not 0.0 <= self.audio_dependency_advantage_weight_min <= 1.0 <= self.audio_dependency_advantage_weight_max:
            raise ValueError(
                "audio dependency advantage weight bounds must satisfy 0 <= min <= 1 <= max"
            )
        if self.audio_dependency_reward and self.num_generations < 2:
            raise ValueError("Audio-dependency reward requires num_generations >= 2 for group normalization")
        if (
            self.audio_dependency_full_vocab_trajectory_advantage
            or self.audio_dependency_full_vocab_js_trajectory_advantage
            or self.audio_dependency_entropy_delta_trajectory_advantage
        ):
            if self.num_generations < 2:
                raise ValueError("Trajectory audio advantage requires num_generations >= 2")
            if not 0.0 <= self.audio_dependency_trajectory_weight_min <= 1.0:
                raise ValueError("Trajectory audio advantage minimum weight must be in [0, 1]")
            if self.audio_dependency_trajectory_weight_max < 1.0:
                raise ValueError("Trajectory audio advantage maximum weight must be >= 1")
        if not 0.0 <= self.audio_dependency_silence_ratio <= 1.0:
            raise ValueError("Audio dependency silence ratio must be in [0, 1]")
        if self.audio_dependency_full_vocab_token_weight_advantage:
            if not 0.0 <= self.audio_dependency_trajectory_token_weight_min <= 1.0:
                raise ValueError("Within-trajectory audio token minimum weight must be in [0, 1]")
            if self.audio_dependency_trajectory_token_weight_max < 1.0:
                raise ValueError("Within-trajectory audio token maximum weight must be >= 1")
        if (
            self.audio_dependency_full_vocab_token_weight_advantage
            and self.audio_dependency_full_vocab_entropy_token_advantage
        ):
            raise ValueError("Choose either audio-only or audio-plus-entropy within-trajectory token weighting")
        enabled_within_trajectory_modes = sum((
            self.audio_dependency_full_vocab_token_weight_advantage,
            self.audio_dependency_full_vocab_entropy_token_advantage,
            self.audio_dependency_full_vocab_js_token_advantage,
            self.audio_dependency_full_vocab_binary_token_advantage,
        ))
        if enabled_within_trajectory_modes > 1:
            raise ValueError("Choose only one within-trajectory token-weighting mode")
        if self.reasoning_entropy_trajectory_advantage:
            if self.num_generations < 2:
                raise ValueError("Reasoning-entropy trajectory weighting requires num_generations >= 2")
            if not 0.0 <= self.reasoning_entropy_trajectory_weight_min <= 1.0:
                raise ValueError("Reasoning-entropy minimum weight must be in [0, 1]")
            if self.reasoning_entropy_trajectory_weight_max < 1.0:
                raise ValueError("Reasoning-entropy maximum weight must be >= 1")
        if not 0.0 < self.audio_dependency_top_token_fraction <= 1.0:
            raise ValueError("Audio dependency top token fraction must be in (0, 1]")

        self.accuracy_reward_index = next(
            (
                i
                for i, reward_func in enumerate(self.reward_funcs)
                if not isinstance(reward_func, PreTrainedModel)
                and getattr(reward_func, "__name__", "") == "accuracy_reward"
            ),
            None,
        )
        if self.audio_dependency_reward and self.accuracy_reward_index is None:
            raise ValueError("Audio-dependency reward requires accuracy_reward for correctness gating")

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        set_seed(args.seed, device_specific=True)

        ref_model_is_device_mapped = False
        if deferred_ref_model_id is not None:
            # The trainable model is CPU/NVMe-offloaded. Keep the frozen
            # reference model directly on the roomy GPU so it does not consume
            # the same constrained host-memory pool. Temporarily disable the
            # Transformers ZeRO-3 loading hook for this one model.
            import transformers.integrations.deepspeed as hf_deepspeed

            saved_hf_ds_config = (
                hf_deepspeed._hf_deepspeed_config_weak_ref()
                if hf_deepspeed._hf_deepspeed_config_weak_ref is not None
                else None
            )
            hf_deepspeed.unset_hf_deepspeed_config()
            try:
                ref_load_kwargs = {
                    **model_init_kwargs,
                    "low_cpu_mem_usage": True,
                    "device_map": {"": self.accelerator.device},
                }
                if "Qwen2-Audio" in deferred_ref_model_id:
                    self.ref_model = Qwen2AudioForConditionalGeneration.from_pretrained(
                        deferred_ref_model_id, **ref_load_kwargs
                    )
                elif "Qwen2.5-Omni" in deferred_ref_model_id:
                    self.ref_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                        deferred_ref_model_id, attn_implementation=None, **ref_load_kwargs
                    )
                else:
                    self.ref_model = AutoModelForCausalLM.from_pretrained(
                        deferred_ref_model_id, **ref_load_kwargs
                    )
            finally:
                if saved_hf_ds_config is not None:
                    hf_deepspeed.set_hf_deepspeed_config(saved_hf_ds_config)
            self.ref_model.requires_grad_(False)
            self.ref_model.eval()
            ref_model_is_device_mapped = True

        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            num_return_sequences=self.num_generations,
            pad_token_id=processing_class.pad_token_id,
            eos_token_id=processing_class.eos_token_id,
        )
        self.answer_end_token_ids = processing_class.tokenizer.encode(
            "</answer>", add_special_tokens=False
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if ref_model_is_device_mapped:
                pass
            elif self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(
        self,
        model,
        input_ids,
        attention_mask,
        features_values,
        features_masks,
        completion_start=None,
    ):
        logits = model(input_ids, attention_mask=attention_mask, input_features=features_values, feature_attention_mask=features_masks).logits  # (B, L, V)
        #logits = model.generate(input_ids, attention_mask=attention_mask, input_features=features_values, feature_attention_mask=features_masks).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        token_logps = selective_log_softmax(logits, input_ids)
        if completion_start is None:
            return token_logps
        return token_logps, logits[:, completion_start:].detach()

    def _get_completion_logits(
        self, model, input_ids, attention_mask, features_values, features_masks, completion_start
    ):
        logits = model(
            input_ids,
            attention_mask=attention_mask,
            input_features=features_values,
            feature_attention_mask=features_masks,
        ).logits[:, :-1, :]
        # Materialize only completion positions so the much larger prompt
        # logits storage can be released before the full-vocabulary KL loop.
        return logits[:, completion_start:].detach().contiguous()

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompts = [x["prompt"] for x in inputs]

        if "Qwen2-Audio" in self.model_id: # for Qwen2-Audio
            if self.audio_dependency_enabled:
                raise NotImplementedError(
                    "Audio-dependency training currently supports Qwen2.5-Omni only"
                )
            prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
            audios = [x["audio"] for x in inputs]
            prompt_inputs = self.processing_class(
                text=prompts_text,
                audios=audios,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
        else: # for Qwen2.5-Omni
            messages = [example["prompt"] for example in inputs]
            prompts_text = self.processing_class.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
            prompt_inputs = self.processing_class(text=prompts_text, images=images, videos=videos, audio=audios, padding=True, return_tensors="pt")

            silent_prompt_inputs = None
            if self.audio_dependency_enabled:
                silent_audios = [
                    randomly_silence_audio(audio, self.audio_dependency_silence_ratio)
                    for audio in audios
                ]
                silent_prompt_inputs = self.processing_class(
                    text=prompts_text,
                    images=images,
                    videos=videos,
                    audio=silent_audios,
                    padding=True,
                    return_tensors="pt",
                )
                if not torch.equal(prompt_inputs["input_ids"], silent_prompt_inputs["input_ids"]):
                    raise ValueError(
                        "Silent and audible prompts produced different token IDs; "
                        "teacher-forced trajectories would not be aligned"
                    )

        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]


        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            answer_stopping_criteria = StoppingCriteriaList(
                [
                    DecodedStringStoppingCriteria(
                        self.processing_class.tokenizer,
                        "</answer>",
                        prompt_ids.size(1),
                    )
                ]
            )
            prompt_completion_ids = unwrapped_model.generate(
                **prompt_inputs,
                generation_config=self.generation_config,
                stopping_criteria=answer_stopping_criteria,
            )
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)


        # Mask everything after the first EOS or the complete </answer> stop
        # sequence. StopStringCriteria pads finished rows while other rows in
        # the batch continue, so EOS-only masking would include those pads.
        device = self.accelerator.device
        completion_mask = completion_mask_through_stop(
            completion_ids,
            self.processing_class.eos_token_id,
            self.processing_class.pad_token_id,
            self.answer_end_token_ids,
        )

        features_values = prompt_inputs["input_features"]
        features_masks = prompt_inputs["feature_attention_mask"]
        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        features_values = features_values.repeat_interleave(self.num_generations, dim=0)
        features_masks = features_masks.repeat_interleave(self.num_generations, dim=0)

        audio_completion_logits = None
        if self.audio_dependency_full_vocab_enabled:
            per_token_logps, audio_completion_logits = self._get_per_token_logps(
                model,
                prompt_completion_ids,
                attention_mask,
                features_values,
                features_masks,
                completion_start=prompt_length - 1,
            )
        else:
            per_token_logps = self._get_per_token_logps(
                model, prompt_completion_ids, attention_mask, features_values, features_masks
            )
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]

        audio_dependency = None
        reasoning_entropy = None
        if self.audio_dependency_enabled:
            silent_prompt_inputs = super()._prepare_inputs(silent_prompt_inputs)
            silent_features_values = silent_prompt_inputs["input_features"].repeat_interleave(
                self.num_generations, dim=0
            )
            silent_features_masks = silent_prompt_inputs["feature_attention_mask"].repeat_interleave(
                self.num_generations, dim=0
            )
            if silent_features_values.shape != features_values.shape or silent_features_masks.shape != features_masks.shape:
                raise ValueError(
                    "Silent and audible inputs produced different feature shapes; "
                    "the silent waveform must preserve the original duration"
                )
            text_token_mask = completion_mask * completion_ids.ne(self.processing_class.eos_token_id)
            if self.audio_dependency_full_vocab_enabled:
                with torch.inference_mode():
                    silent_completion_logits = self._get_completion_logits(
                        model,
                        prompt_completion_ids,
                        attention_mask,
                        silent_features_values,
                        silent_features_masks,
                        completion_start=prompt_length - 1,
                    )
                    if (
                        self.audio_dependency_full_vocab_js_token_advantage
                        or self.audio_dependency_full_vocab_js_trajectory_advantage
                    ):
                        token_audio_dependency = full_vocab_jensen_shannon_divergence(
                            audio_completion_logits, silent_completion_logits
                        )
                    elif self.audio_dependency_entropy_delta_trajectory_advantage:
                        token_audio_dependency = full_vocab_absolute_entropy_delta(
                            audio_completion_logits, silent_completion_logits
                        )
                    else:
                        token_audio_dependency = full_vocab_forward_kl(
                            audio_completion_logits, silent_completion_logits
                        )
                    if (
                        self.reasoning_entropy_trajectory_advantage
                        or self.audio_dependency_full_vocab_entropy_token_advantage
                    ):
                        token_reasoning_entropy = full_vocab_token_entropy(audio_completion_logits)
                denominator = text_token_mask.sum(dim=1).clamp_min(1)
                audio_dependency = (
                    token_audio_dependency * text_token_mask
                ).sum(dim=1) / denominator
                if self.reasoning_entropy_trajectory_advantage:
                    reasoning_entropy = (
                        token_reasoning_entropy * text_token_mask
                    ).sum(dim=1) / denominator
                audio_dependency_clip_fraction = torch.zeros_like(audio_dependency)
                del audio_completion_logits, silent_completion_logits
            else:
                with torch.inference_mode():
                    silent_per_token_logps = self._get_per_token_logps(
                        model,
                        prompt_completion_ids,
                        attention_mask,
                        silent_features_values,
                        silent_features_masks,
                    )
                silent_per_token_logps = silent_per_token_logps[:, prompt_length - 1 :]
                audio_dependency, token_audio_dependency = k3_audio_dependency(
                    per_token_logps,
                    silent_per_token_logps,
                    text_token_mask,
                    token_kl_max=self.audio_dependency_token_kl_max,
                )
                if self.audio_dependency_token_kl_max > 0:
                    audio_dependency_clip_fraction = (
                        token_audio_dependency.eq(self.audio_dependency_token_kl_max) * text_token_mask
                    ).sum(dim=1) / text_token_mask.sum(dim=1).clamp_min(1)
                else:
                    audio_dependency_clip_fraction = torch.zeros_like(audio_dependency)

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, attention_mask, features_values, features_masks)
            else:
                if self.beta == 0:
                    ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, features_values, features_masks)
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, features_values, features_masks)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        correctness = None
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                # Mixed-source batches do not necessarily share every metadata
                # key (for example, AVQA has `video_name` while MMSU does not).
                # Build the union and preserve alignment with a None placeholder.
                reward_keys = set().union(*(example.keys() for example in inputs))
                reward_kwargs = {key: [] for key in reward_keys if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example.get(key)] * self.num_generations)
                output_reward_func = reward_func(prompts=prompts, completions=completions, think=self.think, **reward_kwargs)
                raw_reward = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
                if i == self.accuracy_reward_index:
                    correctness = raw_reward
                rewards_per_func[:, i] = raw_reward * self.reward_weights[i].to(device)

        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)
        audio_reward = None
        normalized_audio_dependency = None
        audio_dependency_group_range = None
        if self.audio_dependency_enabled:
            audio_dependency_group_range = (
                audio_dependency.view(-1, self.num_generations).max(dim=1).values
                - audio_dependency.view(-1, self.num_generations).min(dim=1).values
            )
        if self.audio_dependency_reward:
            audio_reward, normalized_audio_dependency = correctness_gated_audio_reward(
                dependency=audio_dependency,
                correctness=correctness,
                num_generations=self.num_generations,
                reward_lambda=self.audio_dependency_lambda,
                reward_mu=self.audio_dependency_mu,
            )
            rewards = rewards + audio_reward
        #rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        token_advantages = advantages.unsqueeze(1)
        correctness_probability = None
        correctness_uncertainty_weight = None
        if self.correctness_uncertainty_advantage:
            if correctness is None:
                raise ValueError("Correctness uncertainty advantage requires an accuracy reward")
            correctness_probability = correctness.detach().view(-1, self.num_generations).mean(dim=1)
            correctness_uncertainty_weight = 4.0 * correctness_probability * (1.0 - correctness_probability)
            expanded_uncertainty_weight = correctness_uncertainty_weight.repeat_interleave(
                self.num_generations, dim=0
            )
            token_advantages = (advantages * expanded_uncertainty_weight).unsqueeze(1)
        audio_advantage_weights = None
        trajectory_audio_weights = None
        trajectory_audio_normalized = None
        trajectory_entropy_weights = None
        trajectory_entropy_normalized = None
        selected_audio_token_mask = None
        trajectory_token_audio_weights = None
        trajectory_token_audio_normalized = None
        audio_entropy_token_weights = None
        audio_entropy_token_score = None
        token_entropy_normalized = None
        js_token_weights = None
        if (
            self.audio_dependency_full_vocab_trajectory_advantage
            or self.audio_dependency_full_vocab_js_trajectory_advantage
            or self.audio_dependency_entropy_delta_trajectory_advantage
        ):
            if (
                self.audio_dependency_full_vocab_js_token_advantage
                or self.audio_dependency_full_vocab_js_trajectory_advantage
            ):
                trajectory_audio_weights, trajectory_audio_normalized = (
                    minmax_centered_trajectory_advantage_weights(
                        dependency=audio_dependency,
                        num_generations=self.num_generations,
                    )
                )
            else:
                trajectory_audio_weights, trajectory_audio_normalized = minmax_trajectory_advantage_weights(
                    dependency=audio_dependency,
                    num_generations=self.num_generations,
                    weight_min=self.audio_dependency_trajectory_weight_min,
                    weight_max=self.audio_dependency_trajectory_weight_max,
                )
            token_advantages = (advantages * trajectory_audio_weights).unsqueeze(1)
        if self.audio_dependency_full_vocab_token_weight_advantage:
            trajectory_token_audio_weights, trajectory_token_audio_normalized = minmax_within_trajectory_token_weights(
                token_dependency=token_audio_dependency,
                token_mask=text_token_mask,
                weight_min=self.audio_dependency_trajectory_token_weight_min,
                weight_max=self.audio_dependency_trajectory_token_weight_max,
            )
            token_advantages = token_advantages * trajectory_token_audio_weights
        if self.audio_dependency_full_vocab_entropy_token_advantage:
            (
                audio_entropy_token_weights,
                audio_entropy_token_score,
                trajectory_token_audio_normalized,
                token_entropy_normalized,
            ) = audio_entropy_within_trajectory_token_weights(
                token_dependency=token_audio_dependency,
                token_entropy=token_reasoning_entropy,
                token_mask=text_token_mask,
            )
            token_advantages = token_advantages * audio_entropy_token_weights
        if self.audio_dependency_full_vocab_js_token_advantage:
            js_token_weights = minmax_centered_token_weights(
                token_scores=token_audio_dependency,
                token_mask=completion_mask,
            )
            token_advantages = token_advantages * js_token_weights
        if self.audio_dependency_full_vocab_binary_token_advantage:
            trajectory_token_audio_weights = binary_rank_within_trajectory_token_weights(
                token_dependency=token_audio_dependency,
                token_mask=completion_mask,
                low_weight=self.audio_dependency_trajectory_token_weight_min,
                high_weight=self.audio_dependency_trajectory_token_weight_max,
            )
            token_advantages = token_advantages * trajectory_token_audio_weights
        if self.reasoning_entropy_trajectory_advantage:
            trajectory_entropy_weights, trajectory_entropy_normalized = minmax_trajectory_advantage_weights(
                dependency=reasoning_entropy,
                num_generations=self.num_generations,
                weight_min=self.reasoning_entropy_trajectory_weight_min,
                weight_max=self.reasoning_entropy_trajectory_weight_max,
            )
            # Compose the two independent trajectory signals multiplicatively.
            token_advantages = token_advantages * trajectory_entropy_weights.unsqueeze(1)
        if self.audio_dependency_token_advantage:
            token_advantages, audio_advantage_weights = redistribute_advantages_by_audio_dependency(
                advantages=advantages,
                token_dependency=token_audio_dependency,
                token_mask=completion_mask,
                token_kl_max=self.audio_dependency_token_kl_max,
                mix=self.audio_dependency_advantage_mix,
                weight_min=self.audio_dependency_advantage_weight_min,
                weight_max=self.audio_dependency_advantage_weight_max,
            )
        if self.audio_dependency_full_vocab_group_top_token_advantage:
            selected_audio_token_mask = group_top_fraction_token_mask(
                token_dependency=token_audio_dependency,
                token_mask=text_token_mask,
                num_generations=self.num_generations,
                fraction=self.audio_dependency_top_token_fraction,
            )
            # Compose with the trajectory-level weight when both full-vocabulary
            # modes are enabled instead of replacing it.
            token_advantages = token_advantages * selected_audio_token_mask.to(advantages.dtype)
        if self.audio_dependency_full_vocab_trajectory_top_token_advantage:
            selected_audio_token_mask = trajectory_top_fraction_token_mask(
                token_dependency=token_audio_dependency,
                token_mask=text_token_mask,
                fraction=self.audio_dependency_top_token_fraction,
            )
            token_advantages = token_advantages * selected_audio_token_mask.to(advantages.dtype)

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * token_advantages
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        if self.correctness_uncertainty_advantage:
            gathered_probability = self.accelerator.gather_for_metrics(correctness_probability)
            gathered_weight = self.accelerator.gather_for_metrics(correctness_uncertainty_weight)
            self._metrics["correctness_uncertainty/p_correct"].append(gathered_probability.mean().item())
            self._metrics["correctness_uncertainty/weight"].append(gathered_weight.mean().item())
            self._metrics["correctness_uncertainty/active_fraction"].append(
                gathered_weight.gt(0).float().mean().item()
            )

        if self.audio_dependency_enabled:
            self._metrics["audio_dependency/silence_ratio"].append(
                self.audio_dependency_silence_ratio
            )
            gathered_dependency = self.accelerator.gather_for_metrics(audio_dependency)
            self._metrics["audio_dependency/raw"].append(gathered_dependency.mean().item())
            self._metrics["audio_dependency/group_std"].append(
                self.accelerator.gather_for_metrics(
                    audio_dependency.view(-1, self.num_generations).std(dim=1)
                ).mean().item()
            )
            self._metrics["audio_dependency/group_range"].append(
                self.accelerator.gather_for_metrics(audio_dependency_group_range).mean().item()
            )
            self._metrics["audio_dependency/token_clip_fraction"].append(
                self.accelerator.gather_for_metrics(audio_dependency_clip_fraction).mean().item()
            )
            if self.audio_dependency_token_advantage:
                weight_mask = completion_mask.to(audio_advantage_weights.dtype)
                weight_count = weight_mask.sum(dim=1).clamp_min(1.0)
                weight_mean = (audio_advantage_weights * weight_mask).sum(dim=1) / weight_count
                weight_std = torch.sqrt(
                    (((audio_advantage_weights - 1.0) * weight_mask).square().sum(dim=1) / weight_count)
                )
                weight_max = audio_advantage_weights.max(dim=1).values
                weight_min = audio_advantage_weights.masked_fill(
                    ~completion_mask.bool(), torch.inf
                ).min(dim=1).values
                weighted_dependency = (
                    token_audio_dependency * audio_advantage_weights * weight_mask
                ).sum(dim=1) / weight_count
                self._metrics["audio_dependency/adv_weight_std"].append(
                    self.accelerator.gather_for_metrics(weight_std).mean().item()
                )
                self._metrics["audio_dependency/adv_weight_mean"].append(
                    self.accelerator.gather_for_metrics(weight_mean).mean().item()
                )
                self._metrics["audio_dependency/adv_weight_max"].append(
                    self.accelerator.gather_for_metrics(weight_max).mean().item()
                )
                self._metrics["audio_dependency/adv_weight_min"].append(
                    self.accelerator.gather_for_metrics(weight_min).mean().item()
                )
                self._metrics["audio_dependency/adv_weighted"].append(
                    self.accelerator.gather_for_metrics(weighted_dependency).mean().item()
                )
            if (
                self.audio_dependency_full_vocab_trajectory_advantage
                or self.audio_dependency_full_vocab_js_trajectory_advantage
                or self.audio_dependency_entropy_delta_trajectory_advantage
            ):
                self._metrics["audio_dependency/normalized"].append(
                    self.accelerator.gather_for_metrics(trajectory_audio_normalized).mean().item()
                )
                self._metrics["audio_dependency/trajectory_weight_mean"].append(
                    self.accelerator.gather_for_metrics(trajectory_audio_weights).mean().item()
                )
                self._metrics["audio_dependency/trajectory_weight_min"].append(
                    self.accelerator.gather_for_metrics(trajectory_audio_weights).min().item()
                )
                self._metrics["audio_dependency/trajectory_weight_max"].append(
                    self.accelerator.gather_for_metrics(trajectory_audio_weights).max().item()
                )
            if self.reasoning_entropy_trajectory_advantage:
                gathered_entropy = self.accelerator.gather_for_metrics(reasoning_entropy)
                self._metrics["reasoning_entropy/raw"].append(gathered_entropy.mean().item())
                self._metrics["reasoning_entropy/normalized"].append(
                    self.accelerator.gather_for_metrics(trajectory_entropy_normalized).mean().item()
                )
                self._metrics["reasoning_entropy/trajectory_weight_mean"].append(
                    self.accelerator.gather_for_metrics(trajectory_entropy_weights).mean().item()
                )
                self._metrics["reasoning_entropy/trajectory_weight_min"].append(
                    self.accelerator.gather_for_metrics(trajectory_entropy_weights).min().item()
                )
                self._metrics["reasoning_entropy/trajectory_weight_max"].append(
                    self.accelerator.gather_for_metrics(trajectory_entropy_weights).max().item()
                )
                combined_weights = trajectory_audio_weights * trajectory_entropy_weights
                self._metrics["trajectory_weight/combined_mean"].append(
                    self.accelerator.gather_for_metrics(combined_weights).mean().item()
                )
                self._metrics["trajectory_weight/combined_min"].append(
                    self.accelerator.gather_for_metrics(combined_weights).min().item()
                )
                self._metrics["trajectory_weight/combined_max"].append(
                    self.accelerator.gather_for_metrics(combined_weights).max().item()
                )
            if (
                self.audio_dependency_full_vocab_token_weight_advantage
                or self.audio_dependency_full_vocab_binary_token_advantage
            ):
                # Completion sequence lengths may differ across ranks. Reduce to
                # fixed-size scalars locally before distributed gathering.
                token_weight_mask = (
                    completion_mask.bool()
                    if self.audio_dependency_full_vocab_binary_token_advantage
                    else text_token_mask.bool()
                )
                valid_token_weights = trajectory_token_audio_weights.masked_select(token_weight_mask)
                valid_token_scores = token_audio_dependency.masked_select(token_weight_mask)
                if valid_token_weights.numel() > 0:
                    local_weight_min = valid_token_weights.min()
                    local_weight_max = valid_token_weights.max()
                else:
                    # A rank can occasionally receive only EOS/empty
                    # completions. Infinity sentinels keep it out of the
                    # distributed min/max while sums and count remain zero.
                    local_weight_min = torch.tensor(torch.inf, device=device)
                    local_weight_max = torch.tensor(-torch.inf, device=device)
                local_token_stats = torch.stack((
                    valid_token_weights.sum(),
                    torch.tensor(float(valid_token_weights.numel()), device=device),
                    local_weight_min,
                    local_weight_max,
                    valid_token_scores.sum(),
                )).float()
                gathered_token_stats = self.accelerator.gather_for_metrics(local_token_stats).view(-1, 5)
                global_token_count = gathered_token_stats[:, 1].sum().clamp_min(1.0)
                nonempty_ranks = gathered_token_stats[:, 1].gt(0)
                if nonempty_ranks.any():
                    global_weight_min = gathered_token_stats[nonempty_ranks, 2].min()
                    global_weight_max = gathered_token_stats[nonempty_ranks, 3].max()
                else:
                    global_weight_min = torch.tensor(1.0, device=device)
                    global_weight_max = torch.tensor(1.0, device=device)
                self._metrics["audio_dependency/token_weight_mean"].append(
                    (gathered_token_stats[:, 0].sum() / global_token_count).item()
                )
                self._metrics["audio_dependency/token_weight_min"].append(
                    global_weight_min.item()
                )
                self._metrics["audio_dependency/token_weight_max"].append(
                    global_weight_max.item()
                )
                self._metrics["audio_dependency/token_raw"].append(
                    (gathered_token_stats[:, 4].sum() / global_token_count).item()
                )
            if self.audio_dependency_full_vocab_entropy_token_advantage:
                valid = text_token_mask.bool()
                valid_weights = audio_entropy_token_weights.masked_select(valid)
                valid_scores = audio_entropy_token_score.masked_select(valid)
                valid_entropy = token_reasoning_entropy.masked_select(valid)
                if valid_weights.numel() > 0:
                    local_stats = torch.stack((
                        valid_weights.sum(),
                        torch.tensor(float(valid_weights.numel()), device=device),
                        valid_weights.max(),
                        valid_scores.sum(),
                        valid_entropy.sum(),
                    )).float()
                else:
                    local_stats = torch.tensor((0.0, 0.0, 0.0, 0.0, 0.0), device=device)
                gathered_stats = self.accelerator.gather_for_metrics(local_stats).view(-1, 5)
                count = gathered_stats[:, 1].sum().clamp_min(1.0)
                self._metrics["audio_entropy_token/weight_mean"].append(
                    (gathered_stats[:, 0].sum() / count).item()
                )
                self._metrics["audio_entropy_token/weight_max"].append(
                    gathered_stats[:, 2].max().item()
                )
                self._metrics["audio_entropy_token/combined_score_mean"].append(
                    (gathered_stats[:, 3].sum() / count).item()
                )
                self._metrics["audio_entropy_token/raw_entropy"].append(
                    (gathered_stats[:, 4].sum() / count).item()
                )
            if self.audio_dependency_full_vocab_js_token_advantage:
                valid = completion_mask.bool()
                valid_weights = js_token_weights.masked_select(valid)
                valid_scores = token_audio_dependency.masked_select(valid)
                sequence_counts = valid.sum(dim=1).float()
                sequence_weight_sums = (js_token_weights * valid).sum(dim=1)
                sequence_sum_error = (sequence_weight_sums - sequence_counts).abs()
                if valid_weights.numel() > 0:
                    local_stats = torch.stack((
                        valid_weights.sum(),
                        torch.tensor(float(valid_weights.numel()), device=device),
                        valid_weights.min(),
                        valid_weights.max(),
                        valid_scores.sum(),
                        valid_scores.max(),
                        sequence_sum_error.sum(),
                        torch.tensor(float(sequence_sum_error.numel()), device=device),
                    )).float()
                else:
                    local_stats = torch.tensor(
                        (0.0, 0.0, torch.inf, -torch.inf, 0.0, 0.0, 0.0, 0.0), device=device
                    )
                gathered_stats = self.accelerator.gather_for_metrics(local_stats).view(-1, 8)
                count = gathered_stats[:, 1].sum().clamp_min(1.0)
                nonempty_ranks = gathered_stats[:, 1].gt(0)
                sequence_count = gathered_stats[:, 7].sum().clamp_min(1.0)
                self._metrics["audio_js_token/weight_mean"].append(
                    (gathered_stats[:, 0].sum() / count).item()
                )
                self._metrics["audio_js_token/weight_min"].append(
                    gathered_stats[nonempty_ranks, 2].min().item() if nonempty_ranks.any() else 1.0
                )
                self._metrics["audio_js_token/weight_max"].append(
                    gathered_stats[nonempty_ranks, 3].max().item() if nonempty_ranks.any() else 1.0
                )
                self._metrics["audio_js_token/raw_mean"].append(
                    (gathered_stats[:, 4].sum() / count).item()
                )
                self._metrics["audio_js_token/raw_max"].append(
                    gathered_stats[:, 5].max().item()
                )
                self._metrics["audio_js_token/sequence_sum_error"].append(
                    (gathered_stats[:, 6].sum() / sequence_count).item()
                )
            if (
                self.audio_dependency_full_vocab_group_top_token_advantage
                or self.audio_dependency_full_vocab_trajectory_top_token_advantage
            ):
                selected_count = selected_audio_token_mask.sum(dim=1)
                valid_count = text_token_mask.sum(dim=1).clamp_min(1)
                selected_fraction = selected_count / valid_count
                selected_dependency = (
                    token_audio_dependency * selected_audio_token_mask
                ).sum(dim=1) / selected_count.clamp_min(1)
                self._metrics["audio_dependency/selected_token_fraction"].append(
                    self.accelerator.gather_for_metrics(selected_fraction).float().mean().item()
                )
                self._metrics["audio_dependency/selected_token_dependency"].append(
                    self.accelerator.gather_for_metrics(selected_dependency).mean().item()
                )
            if self.audio_dependency_reward:
                gathered_correctness = self.accelerator.gather_for_metrics(correctness)
                self._metrics["audio_dependency/normalized"].append(
                    self.accelerator.gather_for_metrics(normalized_audio_dependency).mean().item()
                )
                self._metrics["audio_dependency/gated_reward"].append(
                    self.accelerator.gather_for_metrics(audio_reward).mean().item()
                )
                correct_mask = gathered_correctness.bool()
                if correct_mask.any():
                    self._metrics["audio_dependency/correct"].append(
                        gathered_dependency[correct_mask].mean().item()
                    )
                if (~correct_mask).any():
                    self._metrics["audio_dependency/incorrect"].append(
                        gathered_dependency[~correct_mask].mean().item()
                    )

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @misc{zeng2026ad2po,
                title  = {Credit Where Audio Matters: Audio-Dependent Credit Assignment for Audio Reasoning},
                author = {Chu Zeng and Pingyi Fan and Wei-Qiang Zhang},
                year   = {2026},
                howpublished = {GitHub repository},
                url    = {https://github.com/shuaijiang/Ke-Omni-R},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=None,
            comet_url=get_comet_experiment_url(),
            trainer_name="AD2PO",
            trainer_citation=citation,
            paper_title="Credit Where Audio Matters: Audio-Dependent Credit Assignment for Audio Reasoning",
            paper_id=None,
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
