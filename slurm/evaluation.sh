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

# python3 scripts/med_eval.py \
#     --model Qwen/Qwen3-0.6B \
#     --seed 1234 \
#     --log_dir model/downloaded/models--Qwen--Qwen3-0.6B/ \
#     --filename med_eval_log.txt
