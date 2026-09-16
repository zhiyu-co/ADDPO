"""Verifiable rewards for multiple-choice audio reasoning."""

import re


ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)
REASONING_FORMAT = re.compile(
    r"<think>.*?</think>\s*<answer>.*?</answer>", flags=re.DOTALL
)


def extract_answer(text: str) -> str:
    match = ANSWER_PATTERN.search(text)
    return (match.group(1) if match else text).strip()


def accuracy_reward(completions, solution, **_):
    """Return one for an exact multiple-choice answer match, otherwise zero."""
    return [
        float(extract_answer(completion[0]["content"]) == extract_answer(reference))
        for completion, reference in zip(completions, solution)
    ]


def format_reward(completions, think=True, **_):
    """Reward a single reasoning block followed by a single answer block."""
    pattern = REASONING_FORMAT if think else re.compile(r"<answer>.*?</answer>", re.DOTALL)
    return [
        float(pattern.fullmatch(completion[0]["content"]) is not None)
        for completion in completions
    ]
