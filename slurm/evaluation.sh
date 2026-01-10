#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhaotagore@gmail.com
#SBATCH -p gpucluster
#SBATCH --job-name=qwen_8b_dense_legal
#SBATCH --output=/Users/918839576/Trepo/Efficient_Domain_Adaptation/logs/qwen_8b_dense_legal_slurm-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:4


cd /Users/918839576/Trepo/Efficient_Domain_Adaptation/
pwd  # This will print the working directory to your log
export PYTHONPATH=/Users/918839576/Trepo/Efficient_Domain_Adaptation:$PYTHONPATH
export WANDB_PROJECT="Efficient_Domain_Adaptation"
export WANDB_DIR="./assets/wandb"

for i in {1..10}; do
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 scripts/med_eval.py \
    --model model/downloaded/qwen3_8b_med_dense \
    --seed $((1234 + i)) \
    --log_dir model/downloaded/qwen3_8b_med_dense/ \
    --filename "med_eval_log_run${i}.txt" \
    --tokenizer Qwen/Qwen3-8B
done

for i in {1..10}; do
  CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 scripts/legal_eval.py \
    --model model/downloaded/qwen3_8b_legal_dense \
    --seed $((1234 + i)) \
    --log_dir model/downloaded/qwen3_8b_legal_dense/ \
    --filename "legal_eval_log_run${i}.txt" \
    --tokenizer Qwen/Qwen3-8B
done