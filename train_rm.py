"""
train_rm.py — Bradley-Terry Reward Model training on PKU-SafeRLHF.

Uses TRL's RewardTrainer with LoRA for memory-efficient fine-tuning of
a Llama-3.2-1B base model. The trained RM provides the scalar reward
signal for the PPO loop in train_ppo.py.

Usage:
    python train_rm.py --config configs/baseline_b.yaml
"""

from __future__ import annotations

import argparse
import os

import torch
import yaml
from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _get_device_setup():
    if torch.cuda.is_available():
        return "cuda", "auto", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", {"": "mps"}, torch.bfloat16
    return "cpu", {"": "cpu"}, torch.float32
from trl import RewardConfig, RewardTrainer

from data_utils import load_pku_dataset, make_rm_dataset


def build_lora_config(cfg: dict, task_type: TaskType) -> LoraConfig:
    return LoraConfig(
        task_type=task_type,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        # 'score' is the reward head (nn.Linear) — must be fully trained, not LoRA'd
        modules_to_save=["score"],
        bias="none",
    )


def train_rm(cfg: dict) -> None:
    os.makedirs(cfg["rm_output_dir"], exist_ok=True)
    _, device_map, dtype = _get_device_setup()

    # -----------------------------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg["rm_base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # RewardTrainer expects right-padding

    # -----------------------------------------------------------------------
    # Model — Llama base + LoRA + scalar reward head
    # -----------------------------------------------------------------------
    lora_config = build_lora_config(cfg, TaskType.SEQ_CLS)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["rm_base_model"],
        num_labels=1,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    print("Loading PKU-SafeRLHF …")
    train_raw = load_pku_dataset("train", max_samples=cfg.get("rm_train_subset"))
    eval_raw = load_pku_dataset("test", max_samples=cfg.get("rm_eval_subset", 1000))

    train_dataset = make_rm_dataset(train_raw, tokenizer, max_length=cfg["max_length"])
    eval_dataset = make_rm_dataset(eval_raw, tokenizer, max_length=cfg["max_length"])

    print(f"Train size: {len(train_dataset):,}  |  Eval size: {len(eval_dataset):,}")
    # Note: TRL 0.12+ RewardTrainer tokenises chosen/rejected internally.

    # -----------------------------------------------------------------------
    # Training config
    # -----------------------------------------------------------------------
    reward_config = RewardConfig(
        output_dir=cfg["rm_output_dir"],
        num_train_epochs=cfg["rm_num_epochs"],
        per_device_train_batch_size=cfg["rm_batch_size"],
        per_device_eval_batch_size=cfg["rm_batch_size"],
        gradient_accumulation_steps=cfg["rm_grad_accumulation"],
        learning_rate=cfg["rm_lr"],
        warmup_ratio=cfg.get("rm_warmup_ratio", 0.05),
        eval_strategy="steps",
        eval_steps=cfg.get("rm_eval_steps", 500),
        save_strategy="steps",
        save_steps=cfg.get("rm_save_steps", 500),
        save_total_limit=3,
        logging_steps=cfg.get("rm_logging_steps", 50),
        bf16=cfg.get("bf16", torch.cuda.is_available()),
        dataloader_num_workers=4,
        report_to="wandb" if cfg.get("use_wandb") else "none",
        run_name=cfg.get("run_name", "baseline_b") + "_rm",
        remove_unused_columns=False,
        max_length=cfg["max_length"],
    )

    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,   # TRL 0.12+ uses processing_class instead of tokenizer
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
    )

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    print("Starting RM training …")
    trainer.train()

    print(f"Saving RM to {cfg['rm_output_dir']}")
    trainer.save_model(cfg["rm_output_dir"])
    tokenizer.save_pretrained(cfg["rm_output_dir"])
    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Reward Model (Baseline B)")
    parser.add_argument("--config", default="configs/baseline_b.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if cfg.get("use_wandb"):
        import wandb
        wandb.init(
            project=cfg.get("wandb_project", "nlp-rlhf"),
            name=cfg.get("run_name", "baseline_b") + "_rm",
        )

    train_rm(cfg)


if __name__ == "__main__":
    main()
