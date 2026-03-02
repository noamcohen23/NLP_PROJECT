"""
data_utils.py — PKU-SafeRLHF data loading and preprocessing.

Two preprocessing modes:
  - RM:  returns HF Dataset with chosen/rejected token pairs for RewardTrainer.
  - PPO: returns HF Dataset with tokenized prompts for the PPO rollout loop.
"""

from __future__ import annotations

from typing import Optional

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

# 19 harm categories defined in PKU-SafeRLHF
HARM_CATEGORIES = [
    "Animal Abuse",
    "Child Abuse",
    "Controversy",
    "Discrimination",
    "Drug Abuse",
    "Financial Crime",
    "Hate Speech",
    "Human Trafficking",
    "Illegal Weapons",
    "Misinformation Regarding LGBTQ+",
    "Non-Violent Unethical Behavior",
    "Privacy Violation",
    "Self-Harm",
    "Sexually Explicit Content",
    "Terrorism",
    "Unfairness And Discrimination",
    "Violence",
    "White-Collar Crime",
    "Other Harms",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_pku_dataset(split: str = "train", max_samples: Optional[int] = None) -> Dataset:
    """Load PKU-SafeRLHF from HuggingFace Hub."""
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _has_chat_template(tokenizer: PreTrainedTokenizer) -> bool:
    return (
        hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template is not None
    )


def format_prompt(prompt: str, tokenizer: PreTrainedTokenizer) -> str:
    """Apply the model's chat template (or a plain fallback) to a bare prompt."""
    if _has_chat_template(tokenizer):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # Fallback for models without a chat template (e.g. gpt2 in smoke tests)
    return f"User: {prompt}\nAssistant:"


def format_prompt_response(
    prompt: str, response: str, tokenizer: PreTrainedTokenizer
) -> str:
    """Apply the model's chat template (or plain fallback) to a (prompt, response) pair."""
    if _has_chat_template(tokenizer):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    return f"User: {prompt}\nAssistant: {response}"


# ---------------------------------------------------------------------------
# RM dataset
# ---------------------------------------------------------------------------

def make_rm_dataset(
    hf_dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    use_safety_preference: bool = False,
) -> Dataset:
    """
    Convert PKU-SafeRLHF into (chosen, rejected) text pairs for TRL RewardTrainer.

    TRL 0.12+ RewardTrainer expects raw text fields `chosen` and `rejected`
    (full prompt+response strings). It handles tokenisation internally.

    Args:
        use_safety_preference: if True, uses `safer_response_id`; otherwise
            uses `better_response_id` (helpfulness quality — correct for Baseline B).
    """
    preference_key = "safer_response_id" if use_safety_preference else "better_response_id"

    def _process(example):
        prompt = example["prompt"]
        better_id = example[preference_key]

        if better_id == 0:
            chosen_resp, rejected_resp = example["response_0"], example["response_1"]
        else:
            chosen_resp, rejected_resp = example["response_1"], example["response_0"]

        return {
            "chosen": format_prompt_response(prompt, chosen_resp, tokenizer),
            "rejected": format_prompt_response(prompt, rejected_resp, tokenizer),
        }

    return hf_dataset.map(
        _process,
        remove_columns=hf_dataset.column_names,
        desc="Formatting RM pairs",
    )


# ---------------------------------------------------------------------------
# PPO dataset
# ---------------------------------------------------------------------------

def make_ppo_dataset(
    hf_dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_prompt_length: int = 256,
) -> Dataset:
    """
    Convert PKU-SafeRLHF into tokenized prompts for PPO rollouts.
    Keeps `query` (raw formatted string) alongside `input_ids` so the
    reward function can use decoded text directly.
    """

    def _process(example):
        query = format_prompt(example["prompt"], tokenizer)
        enc = tokenizer(
            query,
            truncation=True,
            max_length=max_prompt_length,
            padding=False,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "query": query,
        }

    processed = hf_dataset.map(
        _process,
        remove_columns=hf_dataset.column_names,
        desc="Tokenizing PPO prompts",
    )
    processed.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return processed
