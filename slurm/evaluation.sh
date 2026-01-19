#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhaotagore@gmail.com
#SBATCH -p gpucluster
#SBATCH --job-name=eval_qwen3_8b_legal_wanda40_c4
#SBATCH --output=/Users/918839576/Trepo/Efficient_Domain_Adaptation/logs/eval_qwen_8b_legal_wanda40_c4_slurm-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1


cd /Users/918839576/Trepo/Efficient_Domain_Adaptation/
pwd  # This will print the working directory to your log
export PYTHONPATH=/Users/918839576/Trepo/Efficient_Domain_Adaptation:$PYTHONPATH


# for i in {1..10}; do
#   python3 scripts/med_eval.py \
#     --model model/downloaded/qwen3_8b_foresightPBS50_med_c4 \
#     --seed $((1234 + i)) \
#     --log_dir model/downloaded/qwen3_8b_foresightPBS50_med_c4/ \
#     --filename "med_eval_log_run${i}.txt" \
#     --tokenizer Qwen/Qwen3-8B
# done

for i in {5..10}; do
  python3 scripts/legal_eval.py \
    --model model/downloaded/qwen3_8b_legal_wanda40_c4 \
    --seed $((1234 + i)) \
    --log_dir model/downloaded/qwen3_8b_legal_wanda40_c4/ \
    --filename "legal_eval_log_run${i}.txt" \
    --tokenizer Qwen/Qwen3-8B \
    --billsum_batch_size 4
done