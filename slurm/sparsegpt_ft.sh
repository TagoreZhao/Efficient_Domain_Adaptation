#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhaotagore@gmail.com
#SBATCH -p gpucluster
#SBATCH --job-name=llama3_8b_sparsegpt40_legal_c4
#SBATCH --output=/Users/918839576/Trepo/Efficient_Domain_Adaptation/logs/llama3_8b_sparsegpt40_legal_c4_slurm-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:4

cd /Users/918839576/Trepo/Efficient_Domain_Adaptation/
pwd  # This will print the working directory to your log
export PYTHONPATH=/Users/918839576/Trepo/Efficient_Domain_Adaptation:$PYTHONPATH
export WANDB_PROJECT="Efficient_Domain_Adaptation"
export WANDB_DIR="./assets/wandb"

TORCH_DISTRIBUTED_DEBUG=DETAIL \
NCCL_DEBUG=INFO \
torchrun --nproc_per_node=4 scripts/sparsegpt_ft.py \
    --model_name meta-llama/Llama-3.1-8B \
    --tokenizer_name meta-llama/Llama-3.1-8B \
    --model_save_dir model/downloaded/ \
    --run_name llama3_8b_sparsegp40_legal_c4 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 2e-4 \
    --seed 1234 \
    --train_dataset_path data/downloaded/llama3_legal_dataset_train \
    --val_dataset_path data/downloaded/llama3_legal_dataset_val \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --weight_decay 0.01 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --report_to "wandb" \
    --completion_only_loss True \
    --eval_steps 100 \
    --pruning_ratio 0.4 \
    --pruning_nsamples 128 \
    --pruning_seqlen 2048 \
    --pruning_dataset_name c4 \
    --blocksize 128 \
    --percdamp 0.01
