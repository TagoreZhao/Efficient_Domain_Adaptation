# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project for efficient domain adaptation of LLMs using structured pruning (WANDA, SparseGPT, Foresight) combined with LoRA fine-tuning. Targets medical and legal domains using Qwen3 and Llama models.

## Environment Setup

```bash
conda env create -f environment.yml   # creates 'domain_llm' env
conda activate domain_llm
# OR
pip install -r requirements.txt
```

The project root must be on PYTHONPATH for imports to work:
```bash
export PYTHONPATH=/path/to/Efficient_Domain_Adaptation:$PYTHONPATH
```

## Common Commands

### Dataset Construction
```bash
python scripts/construct_med.py    # build medical training data
python scripts/construct_legal.py  # build legal training data
```

### Training (multi-GPU with torchrun)
```bash
# Dense LoRA fine-tuning
torchrun --nproc_per_node=4 scripts/finetuning.py \
    --model_name Qwen/Qwen3-8B --run_name <name> \
    --train_dataset_path data/downloaded/<dataset>_train \
    --val_dataset_path data/downloaded/<dataset>_val

# WANDA pruning + fine-tuning
torchrun --nproc_per_node=4 scripts/wanda_ft.py \
    --model_name Qwen/Qwen3-8B --pruning_ratio 0.5 \
    --pruning_dataset_name c4 ...

# SparseGPT pruning + fine-tuning
torchrun --nproc_per_node=4 scripts/sparsegpt_ft.py \
    --model_name Qwen/Qwen3-8B --pruning_ratio 0.5 \
    --pruning_dataset_name c4 --blocksize 128 --percdamp 0.01 ...

# Foresight pruning + fine-tuning
torchrun --nproc_per_node=4 scripts/foresight_ft.py ...
```

Single-GPU runs: use `python` instead of `torchrun`.

### Evaluation
```bash
# Run 10 evaluation seeds for confidence intervals
for i in {1..10}; do
  python scripts/med_eval.py --model model/downloaded/<model_dir> \
    --seed $((1234 + i)) --log_dir model/downloaded/<model_dir>/ \
    --filename "med_eval_log_run${i}.txt" --tokenizer Qwen/Qwen3-8B
done

# Same pattern for legal_eval.py
```

### SLURM Submission
```bash
sbatch slurm/finetune.sh
sbatch slurm/wanda_ft.sh
sbatch slurm/sparsegpt_ft.sh
sbatch slurm/foresight_ft.sh
```

## Architecture

### Pipeline Flow
1. **Data construction** (`data/`) → 2. **Pruning** (`pruning/`) → 3. **LoRA SFT** (`SFT/`, `scripts/`) → 4. **Evaluation** (`evaluation/`)

### Key Modules

- **`data/`** — Dataset loading and prompt template formatting. `datasets.py` builds mixed-task datasets (PubMedQA, MedNLI, HQS for medical; CASEHOLD, BillSum, ContractNLI for legal). `templates.py` defines per-task prompt formats. `utils.py` converts to HuggingFace prompt/completion format.

- **`pruning/`** — Weight pruning algorithms applied before or during fine-tuning.
  - `prune.py`: `prune_wanda()` (activation-aware one-shot pruning), `prune_sparsegpt()` (Hessian-based second-order pruning), and `foresight_prune()` (mask optimization via gradient signals).
  - `sparsegpt.py`: Core SparseGPT algorithm — Hessian accumulation and block-wise optimal weight update.
  - `WrappedGPT.py`: Hooks into layers to accumulate activation statistics for WANDA calibration.
  - `utils.py`: Layer extraction, calibration data loading (supports C4, WikiText2, medical corpus), sparsity checking.

- **`SFT/`** — Training utilities.
  - `ForesightPruneCallback.py`: TRL callback that applies foresight pruning at configurable epoch boundaries during SFT.
  - `utils.py`: DDP initialization helpers (`rank0_prune_then_sync` ensures pruning happens on rank 0 then broadcasts).

- **`evaluation/`** — Zero-shot evaluation on domain benchmarks with bootstrap confidence intervals (`domain_zero_shot.py`). Perplexity evaluation on WikiText2/Harrison/MultiLegal (`perplexity.py`).

- **`scripts/`** — CLI entry points. Each script uses argparse; all share a common set of LoRA/training args plus method-specific flags (e.g., `--pruning_ratio`, `--pruning_dataset_name`).

### Key Conventions

- Models saved to `model/downloaded/`, datasets to `data/downloaded/` (both gitignored).
- Wandb logging: set `WANDB_PROJECT` and `WANDB_DIR=./assets/wandb` env vars.
- Tokenizer always uses left padding with `pad_token = eos_token`.
- LoRA targets all attention + MLP projections: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.
