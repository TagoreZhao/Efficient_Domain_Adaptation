#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhaotagore@gmail.com
#SBATCH -p gpucluster
#SBATCH --job-name=test_qwen_1.7b_ft
#SBATCH --output=/Users/918839576/Trepo/Efficient_Domain_Adaptation/logs/test_qwen_1.7b_ft_slurm-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:4


cd /Users/918839576/Trepo/Efficient_Domain_Adaptation/
pwd  # This will print the working directory to your log
export PYTHONPATH=/Users/918839576/Trepo/Efficient_Domain_Adaptation:$PYTHONPATH

TORCH_DISTRIBUTED_DEBUG=DETAIL \
NCCL_DEBUG=INFO \
torchrun --nproc_per_node=4 main.py 