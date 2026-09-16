import math

import torch

from trainer.grpo_trainer import (
    full_vocab_jensen_shannon_divergence,
    minmax_centered_token_weights,
    minmax_centered_trajectory_advantage_weights,
)
from dataset.dataset import build_example
from utils.rewards import accuracy_reward, format_reward


def test_jsd_is_symmetric_and_bounded():
    audio = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    silence = torch.tensor([[[0.0, 2.0], [1.0, 1.0]]])
    forward = full_vocab_jensen_shannon_divergence(audio, silence)
    reverse = full_vocab_jensen_shannon_divergence(silence, audio)

    assert torch.allclose(forward, reverse, atol=1e-6)
    assert torch.all(forward >= 0)
    assert torch.all(forward <= math.log(2) + 1e-6)


def test_trajectory_weights_preserve_group_mean():
    dependency = torch.tensor([0.1, 0.2, 0.4, 0.9, 3.0, 3.0, 3.0, 3.0])
    weights, _ = minmax_centered_trajectory_advantage_weights(dependency, num_generations=4)
    grouped = weights.view(-1, 4)

    assert torch.allclose(grouped.mean(dim=1), torch.ones(2))
    assert torch.allclose(grouped[1], torch.ones(4))
    assert torch.all((weights >= 0) & (weights <= 2))


def test_token_weights_preserve_valid_token_sum():
    scores = torch.tensor([[0.1, 0.2, 0.9, 0.0], [2.0, 2.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    weights = minmax_centered_token_weights(scores, mask)

    assert torch.allclose((weights * mask).sum(dim=1), mask.sum(dim=1).float())
    assert torch.allclose(weights[1, :2], torch.ones(2))
    assert torch.all(weights[~mask.bool()] == 0)


def test_prompt_and_rewards_share_the_answer_protocol():
    example = build_example({
        "id": "x",
        "audio_path": "/tmp/x.wav",
        "question_text": "What is audible?",
        "multi_choice": ["speech", "music"],
        "answer": 1,
    })
    completion = [[{"content": "<think>A melody is audible.</think><answer>music</answer>"}]]

    assert "Do not output anything after </answer>." in example["prompt"][0]["content"][1]["text"]
    assert accuracy_reward(completion, [example["solution"]]) == [1.0]
    assert format_reward(completion, think=True) == [1.0]
