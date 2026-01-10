#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhaotagore@gmail.com
#SBATCH -p gpucluster
#SBATCH --job-name=eval_qwen_4b_dense_legal
#SBATCH --output=/Users/918839576/Trepo/Efficient_Domain_Adaptation/logs/eval_qwen_4b_dense_legal_slurm-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1


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


# TORCH_DISTRIBUTED_DEBUG=DETAIL \
# NCCL_DEBUG=INFO \
# torchrun --nproc_per_node=4 scripts/finetuning.py \
#     --model_name Qwen/Qwen3-8B \
#     --tokenizer_name Qwen/Qwen3-8B \
#     --model_save_dir model/downloaded/ \
#     --run_name qwen3_8b_legal_dense \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --learning_rate 2e-4 \
#     --seed 1234 \
#     --train_dataset_path data/downloaded/qwen_legal_dataset_train \
#     --val_dataset_path data/downloaded/qwen_legal_dataset_val \
#     --lora_r 8 \
#     --lora_alpha 16 \
#     --lora_dropout 0.05 \
#     --weight_decay 0.01 \
#     --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
#     --report_to "wandb" \
#     --completion_only_loss True \
#     --eval_steps 100


# for i in {1..10}; do
#   python3 scripts/med_eval.py \
#     --model model/downloaded/qwen3_4b_med_dense \
#     --seed $((1234 + i)) \
#     --log_dir model/downloaded/qwen3_4b_med_dense/ \
#     --filename "med_eval_log_run${i}.txt" \
#     --tokenizer Qwen/Qwen3-4B
# done


# for i in {1..10}; do
#   python3 scripts/legal_eval.py \
#     --model model/downloaded/qwen3_4b_legal_dense \
#     --seed $((1234 + i)) \
#     --log_dir model/downloaded/qwen3_4b_legal_dense/ \
#     --filename "legal_eval_log_run${i}.txt" \
#     --tokenizer Qwen/Qwen3-4B
# done
