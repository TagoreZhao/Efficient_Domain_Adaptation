# EfficientXpert

Code for **EfficientXpert: Efficient Domain Adaptation for Large Language Models via Propagation-Aware Pruning**, by Songlin Zhao, Michael Pitts, and Zhuwei Qin.

Accepted at the [On-Device Intelligence: Foundation Models under Real-World Constraints workshop at NeurIPS 2026](https://odi2026.github.io/).

[Paper (arXiv)](https://arxiv.org/abs/2511.19935) · [PDF](https://arxiv.org/pdf/2511.19935)

EfficientXpert combines propagation-aware ForeSight Mask pruning and Partial Brain Surgeon (PBS) updates for low-rank adapters to adapt language models to legal and medical tasks. This repository also includes Wanda and SparseGPT pruning baselines with LoRA fine-tuning.

The guide below follows the current scripts: **construct datasets → prune and fine-tune → evaluate**. Examples use Qwen3-0.6B and 50% unstructured sparsity as a practical starting point. Refer to the paper for its experiment configurations and results.

## 1. Environment setup

Use Linux with Conda and an NVIDIA GPU with sufficient memory for the selected model, training batches, and pruning calibration. The Wanda and SparseGPT training scripts explicitly use CUDA and load weights in BF16, so use a GPU that supports BF16. The supplied Conda environment specifies Python 3.10 and PyTorch 2.6 with CUDA 12.4 dependencies; the NVIDIA driver must support that runtime.

```bash
git clone https://github.com/TagoreZhao/Efficient_Domain_Adaptation.git
cd Efficient_Domain_Adaptation

conda env create --name domain_llm --file environment.yml
conda activate domain_llm

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p data/downloaded model/downloaded assets logs

MODEL_ID="Qwen/Qwen3-0.6B"
MODEL_TAG="qwen3_0.6b"
RUN_SEED=1234
```

Run all subsequent commands from the repository root in the same Bash session. In a new session, activate the environment and set `PYTHONPATH`, `MODEL_ID`, `MODEL_TAG`, and `RUN_SEED` again.

Use [environment.yml](environment.yml) for the complete training environment. [requirements.txt](requirements.txt) is a separate environment snapshot: it pins CUDA 13 builds of PyTorch and omits training dependencies such as `trl` and `peft`, so installing it alone is insufficient for this workflow.

Models and public datasets are fetched from Hugging Face on first use. For gated models, obtain access and authenticate with `hf auth login`. When changing models, update both `MODEL_ID` and `MODEL_TAG` and reconstruct the datasets with that model's tokenizer/chat template.

## 2. Prepare and construct datasets

### Data sources

| Domain | Task | Dataset | How it is loaded |
| --- | --- | --- | --- |
| Legal | Holding selection | [CaseHOLD](https://huggingface.co/datasets/casehold/casehold) | Hugging Face |
| Legal | Summarization | [BillSum](https://huggingface.co/datasets/FiscalNote/billsum) | Hugging Face |
| Legal | Natural language inference | [ContractNLI](https://huggingface.co/datasets/kiddothe2b/contract-nli) | Hugging Face, `contractnli_a` |
| Medical | Question answering | [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) | Hugging Face, `pqa_artificial` for construction |
| Medical | Natural language inference | [MedNLI](https://physionet.org/content/mednli/1.0.0/) | Local JSONL files |
| Medical | Health question summarization (HQS) | [MeQSum](https://github.com/abachaa/MeQSum) | Local Excel file for construction |
| Medical | Health question summarization (HQS) | [MEDIQA 2021 Task 1](https://github.com/abachaa/MEDIQA2021/tree/main/Task1) | Local Excel file for evaluation |

Before constructing medical datasets, obtain MedNLI through PhysioNet's access process and download MeQSum from its source above. Also obtain the MEDIQA test reference file before medical evaluation. Place the files at these exact paths:

```text
data/downloaded/
├── physionet.org/files/mednli/1.0.0/
│   ├── mli_train_v1.jsonl
│   ├── mli_dev_v1.jsonl
│   └── mli_test_v1.jsonl
├── MeQSum_ACL2019_BenAbacha_Demner-Fushman.xlsx
└── MEDIQA2021-Task1-TestSet-ReferenceSummaries.xlsx
```

The MeQSum spreadsheet must contain `CHQ` and `Summary` columns; the MEDIQA spreadsheet must contain `NLM Question` and `Summary`. These files are not bundled with the repository. The optional `data/downloaded/pubmedqa_maybe_output.json` supplies additional “maybe” examples; it is not required for the default construction settings, which request equal proportions of “yes” and “no”.

### Legal training and validation data

[scripts/construct_legal.py](scripts/construct_legal.py) combines CaseHOLD, BillSum, and ContractNLI, formats examples with the selected tokenizer's chat template, and saves a Hugging Face dataset with `prompt` and `completion` columns.

```bash
python scripts/construct_legal.py \
  --model "$MODEL_ID" \
  --split train \
  --casehold_count 7000 \
  --billsum_count 2000 \
  --contractnli_count 6000 \
  --seed "$RUN_SEED" \
  --output_path "data/downloaded/${MODEL_TAG}_legal_train"

python scripts/construct_legal.py \
  --model "$MODEL_ID" \
  --split validation \
  --casehold_count 500 \
  --billsum_count 100 \
  --contractnli_count 1000 \
  --seed "$RUN_SEED" \
  --output_path "data/downloaded/${MODEL_TAG}_legal_val"
```

The current legal training constructor also streams and appends one C4 example, even though it passes `c4_count=0`. It therefore requires access to [C4](https://huggingface.co/datasets/allenai/c4), and its output includes that additional row.

### Medical training and validation data

The medical constructor is currently named **[scripts/contruct_med.py](scripts/contruct_med.py)** (the filename omits the “s” in “construct”). It combines PubMedQA, MedNLI, and MeQSum and saves the same Hugging Face prompt/completion format.

```bash
python scripts/contruct_med.py \
  --model "$MODEL_ID" \
  --split train \
  --pubmed_count 7000 \
  --mednli_count 7000 \
  --hqs_count 1000 \
  --seed "$RUN_SEED" \
  --output_path "data/downloaded/${MODEL_TAG}_med_train"

python scripts/contruct_med.py \
  --model "$MODEL_ID" \
  --split validation \
  --pubmed_count 500 \
  --mednli_count 1000 \
  --hqs_count 100 \
  --seed "$RUN_SEED" \
  --output_path "data/downloaded/${MODEL_TAG}_med_val"
```

Use `--split validation`, even though the constructor help text says “train/val”. Counts are requested sample counts; available examples and label proportions can affect the actual output size.

**Validation sampling:** CaseHOLD and ContractNLI use their validation splits; BillSum construction uses its `test` split for validation, while final evaluation uses `ca_test`. MedNLI uses its development file. PubMedQA training and validation are sampled from the same `pqa_artificial` split without excluding overlap. MeQSum validation is sampled from the same spreadsheet as training; with all 1,000 examples used for training, the validation examples are also in the training set. These construction scripts do not guarantee disjoint validation data for every task.

## 3. Pruning and LoRA fine-tuning

Choose a domain whose training and validation datasets you constructed:

```bash
DOMAIN="legal"  # Set to "med" for medical adaptation.
```

Run any of the three methods below. To train both domains, set `DOMAIN="med"` and rerun the desired commands after the legal runs. Each method saves its merged model to `model/downloaded/<run_name>`; use a new run name when changing the model, seed, or pruning settings.

All examples use three epochs, learning rate `2e-4`, LoRA rank 8, and completion-only loss. W&B logging is disabled with `--report_to none`.

### Wanda

[scripts/wanda_ft.py](scripts/wanda_ft.py) applies Wanda pruning before LoRA fine-tuning, merges the adapters, and applies Wanda again before saving.

```bash
python scripts/wanda_ft.py \
  --model_name "$MODEL_ID" \
  --tokenizer_name "$MODEL_ID" \
  --model_save_dir model/downloaded \
  --run_name "${MODEL_TAG}_${DOMAIN}_wanda50" \
  --train_dataset_path "data/downloaded/${MODEL_TAG}_${DOMAIN}_train" \
  --val_dataset_path "data/downloaded/${MODEL_TAG}_${DOMAIN}_val" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --learning_rate 2e-4 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --weight_decay 0.01 \
  --completion_only_loss True \
  --eval_steps 100 \
  --seed "$RUN_SEED" \
  --report_to none \
  --pruning_ratio 0.5 \
  --pruning_dataset_name c4 \
  --pruning_nsamples 128 \
  --pruning_seqlen 2048
```

### SparseGPT

[scripts/sparsegpt_ft.py](scripts/sparsegpt_ft.py) applies SparseGPT before LoRA fine-tuning and again after merging the adapters.

```bash
python scripts/sparsegpt_ft.py \
  --model_name "$MODEL_ID" \
  --tokenizer_name "$MODEL_ID" \
  --model_save_dir model/downloaded \
  --run_name "${MODEL_TAG}_${DOMAIN}_sparsegpt50" \
  --train_dataset_path "data/downloaded/${MODEL_TAG}_${DOMAIN}_train" \
  --val_dataset_path "data/downloaded/${MODEL_TAG}_${DOMAIN}_val" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --learning_rate 2e-4 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --weight_decay 0.01 \
  --completion_only_loss True \
  --eval_steps 100 \
  --seed "$RUN_SEED" \
  --report_to none \
  --pruning_ratio 0.5 \
  --pruning_dataset_name c4 \
  --pruning_nsamples 128 \
  --pruning_seqlen 2048 \
  --blocksize 128 \
  --percdamp 0.01
```

### ForeSight / EfficientXpert

[scripts/foresight_ft.py](scripts/foresight_ft.py) uses `ForesightPruneCallback` to prune at the start of each epoch after the first, and once more at training end. PBS updates are enabled by default during the intermediate pruning calls; the final call disables PBS and score averaging before the adapters are merged and the model is saved.

```bash
python scripts/foresight_ft.py \
  --model_name "$MODEL_ID" \
  --tokenizer_name "$MODEL_ID" \
  --model_save_dir model/downloaded \
  --run_name "${MODEL_TAG}_${DOMAIN}_foresight50" \
  --train_dataset_path "data/downloaded/${MODEL_TAG}_${DOMAIN}_train" \
  --val_dataset_path "data/downloaded/${MODEL_TAG}_${DOMAIN}_val" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --learning_rate 2e-4 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --weight_decay 0.01 \
  --completion_only_loss True \
  --eval_steps 100 \
  --seed "$RUN_SEED" \
  --report_to none \
  --prune_ratio 0.5 \
  --mask_lr 0.5 \
  --calib_names c4 \
  --calib_nsamples 128 \
  --calib_seqlen 2048 \
  --skip_last_n_epochs 0
```

| Setting | Wanda / SparseGPT | ForeSight |
| --- | --- | --- |
| Target fraction of pruned weights | `--pruning_ratio` | `--prune_ratio` |
| Calibration corpus | `--pruning_dataset_name` | `--calib_names` |
| Calibration sample count | `--pruning_nsamples` | `--calib_nsamples` |
| Calibration sequence length | `--pruning_seqlen` | `--calib_seqlen` |

`0.5` requests 50% sparsity in the pruned layers. Wanda and SparseGPT default to unstructured pruning (`prune_n=prune_m=0`). Calibration uses C4 separately from the domain fine-tuning datasets. In the current ForeSight script, calibration sampling uses the loader's default seed of 0; `--seed` configures training but is not passed to that loader.

Several CLI booleans use Python's `type=bool`: the string `False` is interpreted as true. Leave false-default options omitted; do not use `--PBS False` to disable PBS. `--skip_last_n_epochs` disables PBS in the specified final epochs, but does not skip the pruning calls themselves.

For multi-GPU training, replace the initial `python` in a training command with `torchrun --standalone --nproc_per_node=4`. For ForeSight, also add `--rank0_only True` so only rank 0 computes pruning updates and broadcasts them. Each GPU must fit a model replica. The [SLURM examples](slurm/) require editing their local paths, partition, GPU request, and logging settings for your cluster.

To enable W&B, change `--report_to none` to `--report_to wandb`, authenticate with `wandb login`, and optionally set `WANDB_PROJECT` and `WANDB_DIR`. Reduce the per-device batch sizes if GPU memory is insufficient.

## 4. Domain evaluation

Evaluate the final merged model directory, using the original base tokenizer explicitly. Choose the method you trained:

```bash
METHOD="foresight"  # Or "wanda" or "sparsegpt".
```

### Legal evaluation

```bash
python scripts/legal_eval.py \
  --model "model/downloaded/${MODEL_TAG}_legal_${METHOD}50" \
  --tokenizer "$MODEL_ID" \
  --seed "$RUN_SEED" \
  --log_dir "model/downloaded/${MODEL_TAG}_legal_${METHOD}50/eval" \
  --filename "legal_seed${RUN_SEED}.txt" \
  --billsum_batch_size 1
```

### Medical evaluation

```bash
python scripts/med_eval.py \
  --model "model/downloaded/${MODEL_TAG}_med_${METHOD}50" \
  --tokenizer "$MODEL_ID" \
  --seed "$RUN_SEED" \
  --log_dir "model/downloaded/${MODEL_TAG}_med_${METHOD}50/eval" \
  --filename "med_seed${RUN_SEED}.txt"
```

Run the evaluation matching the trained domain. Both scripts load evaluation data directly; the constructed validation directory is used by the trainer, not these evaluation commands. Medical evaluation requires the local MedNLI test file and MEDIQA spreadsheet listed above. Perplexity corpora are fetched from Hugging Face.

| Domain | Evaluation data | Metrics |
| --- | --- | --- |
| Legal | [Multi-Legal-Pile](https://huggingface.co/datasets/joelniklaus/Multi_Legal_Pile), first 300 `en_legislation` records | Perplexity |
| Legal | ContractNLI test (up to 1,991 examples), CaseHOLD test (200 examples) | Accuracy, macro-F1 |
| Legal | BillSum `ca_test` (200 examples) | ROUGE-1, ROUGE-2, ROUGE-L |
| Medical | Harrison textbook text from [MedQA corpus](https://huggingface.co/datasets/cogbuji/medqa_corpus_en) | Perplexity |
| Medical | PubMedQA `pqa_labeled` (500 examples), MedNLI test | Accuracy, macro-F1 |
| Medical | MEDIQA 2021 Task 1 (100 examples) | ROUGE-1, ROUGE-2, ROUGE-L |

The legal log labels perplexity as “Legal Case Reports”, although its loader uses `en_legislation`. Both perplexity evaluations use sequence length 2048. The evaluators use one GPU (`cuda:0`) when available and load the entire model onto that device.

Summary metrics are printed and written to `--log_dir/--filename`. Task-level predictions and generated summaries are written to fixed filenames under `assets/`; these are overwritten by later evaluations. Keep that directory present and copy its outputs between runs if you need to retain them. Reusing a summary filename also overwrites that log.

For repeated evaluation with separate summary logs, set `DOMAIN` and `METHOD` to a trained combination and run:

```bash
for EVAL_SEED in {1235..1244}; do
  python "scripts/${DOMAIN}_eval.py" \
    --model "model/downloaded/${MODEL_TAG}_${DOMAIN}_${METHOD}50" \
    --tokenizer "$MODEL_ID" \
    --seed "$EVAL_SEED" \
    --log_dir "model/downloaded/${MODEL_TAG}_${DOMAIN}_${METHOD}50/eval" \
    --filename "${DOMAIN}_seed${EVAL_SEED}.txt"
done
```

This repeats evaluation of one saved model; it does not retrain the model or aggregate results automatically.

## Citation

```bibtex
@article{zhao2025efficientxpert,
  title={EfficientXpert: Efficient Domain Adaptation for Large Language Models via Propagation-Aware Pruning},
  author={Zhao, Songlin and Pitts, Michael and Qin, Zhuwei},
  journal={arXiv preprint arXiv:2511.19935},
  year={2025},
  url={https://arxiv.org/abs/2511.19935}
}
```

## Acknowledgments

The Wanda and SparseGPT pruning code in this repository is adapted from their original implementations:

- **[Wanda](https://github.com/locuslab/wanda)** — *A Simple and Effective Pruning Approach for Large Language Models*.
- **[SparseGPT](https://github.com/IST-DASLab/sparsegpt)** — *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot*.

Please refer to the upstream repositories for their papers, implementation details, and licenses, and cite them when using the corresponding methods. We also thank the creators of the datasets used in these experiments.
