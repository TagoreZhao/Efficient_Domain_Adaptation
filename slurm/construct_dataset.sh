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

# python3 scripts/contruct_med.py \
#     --model Qwen/Qwen3-8B \
#     --output_path data/downloaded/qwen_med_dataset_train \
#     --split train 

# python3 scripts/contruct_med.py \
#     --model Qwen/Qwen3-8B \
#     --output_path data/downloaded/qwen_med_dataset_val \
#     --split validation \
#     --pubmed_count 500 \
#     --mednli_count 1000 \
#     --hqs_count 100 

# python3 scripts/construct_legal.py \
#     --model Qwen/Qwen3-8B \
#     --output_path data/downloaded/qwen_legal_dataset_train \
#     --split train

# python3 scripts/construct_legal.py \
#     --model Qwen/Qwen3-8B \
#     --output_path data/downloaded/qwen_legal_dataset_val \
#     --split validation \
#     --casehold_count 500 \
#     --billsum_count 100 \
#     --contractnli_count 1000