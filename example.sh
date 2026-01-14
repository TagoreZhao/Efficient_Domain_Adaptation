#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhaotagore@gmail.com
#SBATCH -p gpucluster
#SBATCH --job-name=llama2_7b_dense_legal
#SBATCH --output=/Users/918839576/Trepo/Efficient_Domain_Adaptation/logs/llama2_7b_dense_legal_slurm-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:4

cd /Users/918839576/Trepo/Efficient_Domain_Adaptation/
pwd  # This will print the working directory to your log
export PYTHONPATH=/Users/918839576/Trepo/Efficient_Domain_Adaptation:$PYTHONPATH
export WANDB_PROJECT="Efficient_Domain_Adaptation"
export WANDB_DIR="./assets/wandb"

TORCH_DISTRIBUTED_DEBUG=DETAIL \
NCCL_DEBUG=INFO \
torchrun --nproc_per_node=4 scripts/finetuning.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --tokenizer_name meta-llama/Llama-2-7b-hf \
    --model_save_dir model/downloaded/ \
    --run_name llama2_7b_legal_dense \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 2e-4 \
    --seed 1234 \
    --train_dataset_path data/downloaded/qwen_legal_dataset_train \
    --val_dataset_path data/downloaded/qwen_legal_dataset_val \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --weight_decay 0.01 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --report_to "wandb" \
    --completion_only_loss True \
    --eval_steps 100

# TORCH_DISTRIBUTED_DEBUG=DETAIL \
# NCCL_DEBUG=INFO \
# torchrun --nproc_per_node=4 scripts/wanda_ft.py \
#     --model_name Qwen/Qwen3-8B \
#     --tokenizer_name Qwen/Qwen3-8B \
#     --model_save_dir model/downloaded/ \
#     --run_name qwen3_8b_med_wanda40_c4 \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --learning_rate 2e-4 \
#     --seed 1234 \
#     --train_dataset_path data/downloaded/qwen_med_dataset_train \
#     --val_dataset_path data/downloaded/qwen_med_dataset_val \
#     --lora_r 8 \
#     --lora_alpha 16 \
#     --lora_dropout 0.05 \
#     --weight_decay 0.01 \
#     --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
#     --report_to "wandb" \
#     --completion_only_loss True \
#     --eval_steps 100 \
#     --pruning_ratio 0.4 \
#     --pruning_nsamples 128 \
#     --pruning_seqlen 2048 \
#     --pruning_dataset_name c4 \

# for i in {1..10}; do
#   python3 scripts/med_eval.py \
#     --model model/downloaded/qwen3_8b_med_wanda60_c4 \
#     --seed $((1234 + i)) \
#     --log_dir model/downloaded/qwen3_8b_med_wanda60_c4/ \
#     --filename "med_eval_log_run${i}.txt" \
#     --tokenizer Qwen/Qwen3-8B
# done

# for i in {1..10}; do
#   python3 scripts/legal_eval.py \
#     --model model/downloaded/qwen3_4b_legal_wanda60_c4 \
#     --seed $((1234 + i)) \
#     --log_dir model/downloaded/qwen3_4b_legal_wanda60_c4/ \
#     --filename "legal_eval_log_run${i}.txt" \
#     --tokenizer Qwen/Qwen3-4B
# done