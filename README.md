# Defense Against the Dark Chat — Baseline B: Vanilla RLHF (RM Only)

This repository implements **Baseline B** for the project *Category Controllable Safety in RLHF via an Encoder Based Harm Signal*.

Baseline B is a standard PPO fine-tuning pipeline where the only reward signal is a preference-based Reward Model (RM) trained on [PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF). No safety encoder is used. This serves as the comparison point for our main method.

```
PKU-SafeRLHF dataset
        │
        ▼
[train_rm.py]   — Bradley-Terry RM (helpfulness preference)
        │
        ▼
[train_ppo.py]  — PPO fine-tuning with RM-only reward
        │
        ▼
[evaluate.py]   — harm rate · quality (RM score) · refusal rate
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone git@github.com:noamcohen23/NLP_PROJECT.git
cd NLP_PROJECT
pip install -r requirements.txt
```

### 2. Authenticate with HuggingFace (required for Llama)

Llama-3.2-1B requires accepting Meta's license on the [HuggingFace model page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) and then logging in:

```bash
huggingface-cli login
```

### 3. (Optional) Set up Weights & Biases

```bash
wandb login
```

Then set `use_wandb: true` in your config file.

---

## Running the Full Pipeline

All scripts take a `--config` argument pointing to a YAML file. Two configs are provided:

| Config | Purpose |
|---|---|
| `configs/baseline_b.yaml` | Full run on GPU with Llama-3.2-1B |
| `configs/smoke_test.yaml` | Quick CPU sanity check with gpt2 |

### Stage 1 — Train the Reward Model

```bash
python train_rm.py --config configs/baseline_b.yaml
```

Trains a Bradley-Terry preference RM on PKU-SafeRLHF using the `better_response_id` (helpfulness) preference signal. Uses LoRA on `meta-llama/Llama-3.2-1B`.

- Output: `outputs/rm/`
- Expected time: ~2–4 hours on L40S (full dataset)

### Stage 2 — PPO Fine-tuning

```bash
python train_ppo.py --config configs/baseline_b.yaml
```

Fine-tunes `meta-llama/Llama-3.2-1B-Instruct` via PPO using the trained RM as the sole reward signal. A KL penalty against the frozen reference model prevents reward hacking.

- Output: `outputs/ppo_baseline_b/`
- Expected time: ~4–8 hours on L40S (320k episodes)

> **Note:** Stage 2 requires Stage 1 to be complete (`outputs/rm/` must exist).

### Stage 3 — Evaluate

```bash
# In-distribution only (PKU-SafeRLHF test set)
python evaluate.py --config configs/baseline_b.yaml

# Evaluate a specific checkpoint
python evaluate.py --config configs/baseline_b.yaml \
                   --model-path outputs/ppo_baseline_b/checkpoint-2000

# Also evaluate on an OOD dataset
python evaluate.py --config configs/baseline_b.yaml \
                   --ood-dataset allenai/real-toxicity-prompts
```

Metrics reported:
- **mean_rm_score** — average reward model score (quality proxy)
- **harm_rate** — fraction of responses classified as harmful
- **refusal_rate** — fraction of responses that refuse to answer (collapse check)

Results are saved to `outputs/eval_baseline_b/metrics.json` and `results.json`.

---

## Smoke Test (CPU, no GPU required)

Runs the full pipeline with `gpt2` on 30 examples to verify the code works before launching on the cluster:

```bash
python train_rm.py  --config configs/smoke_test.yaml
python train_ppo.py --config configs/smoke_test.yaml
python evaluate.py  --config configs/smoke_test.yaml \
                    --model-path outputs/smoke_ppo
```

Takes ~2 minutes total on CPU.

---

## Configuration

All hyperparameters live in `configs/baseline_b.yaml`. Key settings:

```yaml
# Models
policy_model: "meta-llama/Llama-3.2-1B-Instruct"
rm_base_model: "meta-llama/Llama-3.2-1B"

# To speed up a dev run, limit the dataset size:
rm_train_subset: 5000     # null = full dataset (~73k)
ppo_train_subset: 2000    # null = full dataset (~73k)

# Enable W&B logging:
use_wandb: true
wandb_project: "nlp-rlhf"
run_name: "baseline_b"
```

---

## Project Structure

```
NLP_PROJECT/
├── configs/
│   ├── baseline_b.yaml     # Full GPU run (Llama-3.2-1B)
│   └── smoke_test.yaml     # Quick CPU check (gpt2)
├── data_utils.py           # PKU-SafeRLHF loading & preprocessing
├── train_rm.py             # Stage 1: Reward Model training
├── train_ppo.py            # Stage 2: PPO fine-tuning
├── evaluate.py             # Stage 3: Evaluation
└── requirements.txt
```

---

## GPU Requirements

| Model | VRAM needed |
|---|---|
| Llama-3.2-1B (PPO: policy + ref + value + RM) | ~10 GB |
| Llama-2-7B (PPO: policy + ref + value + RM) | ~42 GB |

Tested on: L40S (46 GB), RTX 8000 (46 GB).

To switch to Llama-2-7B, update `configs/baseline_b.yaml`:
```yaml
policy_model: "meta-llama/Llama-2-7b-chat-hf"
rm_base_model: "meta-llama/Llama-2-7b-hf"
```

---

## Troubleshooting

**OOM during PPO training**
Reduce `ppo_per_device_batch_size` (e.g. `32`) and `ppo_local_rollout_forward_batch_size` (e.g. `16`).

**`huggingface_hub.errors.RepositoryNotFoundError`**
Run `huggingface-cli login` and accept the Llama license on the HF model page.

**SIGBUS on macOS (Apple Silicon)**
Already handled in `evaluate.py` via `OMP_NUM_THREADS=1` and weight cloning. Not an issue on Linux GPU servers.

**TRL version mismatch**
This code targets TRL 0.24. Check with `pip show trl`. If you need to pin: `pip install trl==0.24.0`.
