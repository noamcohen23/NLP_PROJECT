"""
train_ppo.py — Vanilla RLHF (Baseline B): PPO fine-tuning with RM-only reward.

TRL 0.12+ / 0.24 API:
  - PPOTrainer takes model, ref_model, reward_model, and value_model as
    separate arguments and handles generation + scoring internally.
  - Training is triggered by trainer.train() — no manual rollout loop needed.

Pipeline:
  1. Load the frozen RM trained by train_rm.py as reward_model.
  2. Initialize value_model from the same policy architecture.
  3. Fine-tune the policy via PPO; KL penalty against ref_model prevents hacking.

Usage:
    python train_ppo.py --config configs/baseline_b.yaml
"""

from __future__ import annotations

import argparse
import copy
import os

import torch
import yaml
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from trl import PPOConfig, PPOTrainer

from data_utils import load_pku_dataset, make_ppo_dataset


def train_ppo(cfg: dict) -> None:
    os.makedirs(cfg["ppo_output_dir"], exist_ok=True)

    # -----------------------------------------------------------------------
    # Tokenizer (shared by policy and reference models)
    # -----------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg["policy_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # required for generation batching

    # -----------------------------------------------------------------------
    # Policy model (trainable)
    # -----------------------------------------------------------------------
    print("Loading policy model …")
    policy_model = AutoModelForCausalLM.from_pretrained(
        cfg["policy_model"],
        torch_dtype=torch.bfloat16,
    )

    # -----------------------------------------------------------------------
    # Reference model (frozen copy of SFT policy for KL penalty)
    # -----------------------------------------------------------------------
    print("Loading reference model …")
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg["policy_model"],
        torch_dtype=torch.bfloat16,
    )
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()

    # -----------------------------------------------------------------------
    # Value model — same architecture as policy, used as critic for GAE.
    # Initialized from the policy model weights (standard PPO practice).
    # -----------------------------------------------------------------------
    print("Loading value model …")
    value_model = AutoModelForSequenceClassification.from_pretrained(
        cfg["policy_model"],
        num_labels=1,
        torch_dtype=torch.bfloat16,
    )
    # value_model is trained jointly with the policy — keep grads on

    # -----------------------------------------------------------------------
    # Reward model (frozen RM trained by train_rm.py)
    # -----------------------------------------------------------------------
    print("Loading reward model …")
    rm_tokenizer = AutoTokenizer.from_pretrained(cfg["rm_output_dir"])
    if rm_tokenizer.pad_token is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        cfg["rm_output_dir"],
        num_labels=1,
        torch_dtype=torch.bfloat16,
    )
    for p in reward_model.parameters():
        p.requires_grad_(False)
    reward_model.eval()

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    print("Loading PPO dataset …")
    train_raw = load_pku_dataset("train", max_samples=cfg.get("ppo_train_subset"))
    ppo_dataset = make_ppo_dataset(
        train_raw, tokenizer, max_prompt_length=cfg["max_prompt_length"]
    )
    # PPOTrainer's DataCollatorWithPadding expects only input_ids/attention_mask
    ppo_dataset = ppo_dataset.remove_columns(
        [c for c in ppo_dataset.column_names if c not in ("input_ids", "attention_mask")]
    )
    print(f"PPO dataset size: {len(ppo_dataset):,}")

    # Small eval split for PPOTrainer's generate_completions (required by TRL 0.24)
    eval_raw = load_pku_dataset("test", max_samples=cfg.get("ppo_eval_subset", 16))
    ppo_eval_dataset = make_ppo_dataset(
        eval_raw, tokenizer, max_prompt_length=cfg["max_prompt_length"]
    )
    ppo_eval_dataset = ppo_eval_dataset.remove_columns(
        [c for c in ppo_eval_dataset.column_names if c not in ("input_ids", "attention_mask")]
    )
    print(f"PPO eval dataset size: {len(ppo_eval_dataset):,}")

    # -----------------------------------------------------------------------
    # LoRA config (applied to the policy by PPOTrainer when peft_config is set)
    # -----------------------------------------------------------------------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias="none",
    )

    # -----------------------------------------------------------------------
    # PPO config  (TRL 0.24 — TrainingArguments-based)
    # -----------------------------------------------------------------------
    ppo_config = PPOConfig(
        output_dir=cfg["ppo_output_dir"],
        # Rollout / update batch sizes
        per_device_train_batch_size=cfg["ppo_per_device_batch_size"],
        num_mini_batches=cfg["ppo_num_mini_batches"],
        local_rollout_forward_batch_size=cfg.get(
            "ppo_local_rollout_forward_batch_size",
            cfg["ppo_per_device_batch_size"],
        ),
        # Episode / step budget
        total_episodes=cfg.get("ppo_total_episodes", 50_000),
        # PPO hyperparameters
        num_ppo_epochs=cfg["ppo_epochs"],
        kl_coef=cfg["ppo_kl_coeff"],
        cliprange=cfg.get("ppo_cliprange", 0.2),
        cliprange_value=cfg.get("ppo_cliprange_value", 0.2),
        vf_coef=cfg.get("ppo_vf_coef", 0.1),
        # Generation
        response_length=cfg["max_new_tokens"],
        temperature=cfg.get("temperature", 0.7),
        # Optimisation
        learning_rate=cfg["ppo_lr"],
        max_grad_norm=cfg.get("ppo_max_grad_norm", 1.0),
        gradient_accumulation_steps=cfg.get("ppo_gradient_accumulation_steps", 1),
        warmup_ratio=cfg.get("ppo_warmup_ratio", 0.05),
        # Logging / saving
        logging_steps=cfg.get("ppo_logging_steps", 10),
        save_steps=cfg.get("ppo_save_steps", 500),
        save_total_limit=3,
        bf16=cfg.get("bf16", True),
        report_to="wandb" if cfg.get("use_wandb") else "none",
        run_name=cfg.get("run_name", "baseline_b") + "_ppo",
        # Misc
        remove_unused_columns=False,
        sft_model_path=cfg["policy_model"],   # logged to run metadata
        reward_model_path=cfg["rm_output_dir"],
    )

    # -----------------------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------------------
    trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=policy_model,
        ref_model=ref_model,
        reward_model=reward_model,
        train_dataset=ppo_dataset,
        eval_dataset=ppo_eval_dataset,
        value_model=value_model,
        peft_config=lora_config,
    )

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    print("Starting PPO training …")
    trainer.train()

    print(f"Saving PPO model to {cfg['ppo_output_dir']}")
    trainer.save_model(cfg["ppo_output_dir"])
    tokenizer.save_pretrained(cfg["ppo_output_dir"])
    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PPO Training — Baseline B (RM only)")
    parser.add_argument("--config", default="configs/baseline_b.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if cfg.get("use_wandb"):
        import wandb
        wandb.init(
            project=cfg.get("wandb_project", "nlp-rlhf"),
            name=cfg.get("run_name", "baseline_b") + "_ppo",
        )

    train_ppo(cfg)


if __name__ == "__main__":
    main()
