import torch
from transformers import PreTrainedTokenizerBase, StoppingCriteria


class DecodedStringStoppingCriteria(StoppingCriteria):
    """Stop each generated row after a decoded string appears after the prompt."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, stop_string: str, prompt_length: int):
        if not stop_string:
            raise ValueError("stop_string must not be empty")
        self.tokenizer = tokenizer
        self.stop_string = stop_string
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        generated_ids = input_ids[:, self.prompt_length :]
        if generated_ids.size(1) == 0:
            return torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return torch.tensor(
            [self.stop_string in text for text in generated_text],
            dtype=torch.bool,
            device=input_ids.device,
        )
