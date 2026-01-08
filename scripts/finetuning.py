import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.domain_zero_shot import *
from evaluation.perplexity import eval_ppl
from trl import  SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model
from data.datasets import construct_med_data
from data.utils import to_prompt_completion_hf


def parse_args():

    parser = argparse.ArgumentParser(description="Fine-tune a language model with LoRA and SFT.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B", help="Pre-trained model name or path.")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Tokenizer name or path. If not provided, model_name will be used.")
    parser.add_argument("--model_save_dir", type=str, default="model/downloaded", help="Directory to save the model.")
    parser.add_argument("--run_name", type=str, default="there_is_no_name", help="Run name for logging.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Training batch size per device.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Evaluation batch size per device.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--eval_steps", type=int, default=100, help="Number of steps between evaluations.")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate.")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,o_proj,v_proj,k_proj,gate_proj,up_proj,down_proj", help="Comma-separated list of target modules for LoRA.")
    parser.add_argument("--train_dataset_path", type=str, default='data/downloaded/test_medical_data', help="Path to the training dataset.")
    parser.add_argument("--val_dataset_path", type=str, default='data/downloaded/test_medical_data', help="Path to the validation dataset.")
    parser.add_argument("--report_to", type=str, default="wandb", help="Reporting tool for logging.")
    parser.add_argument("--ddp_find_unused_parameters", type=bool, default=False, help="DDP find unused parameters flag.")
    parser.add_argument("--completion_only_loss", type=bool, default=False, help="Use completion only loss.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay for optimizer.")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    torch.cuda.empty_cache()

    peft_model_save_dir = os.path.join(args.model_save_dir, args.run_name)
    peft_config = LoraConfig(
        r = args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none" ,
        target_modules=args.lora_target_modules.split(","),
    )

    sft_config = SFTConfig(
        output_dir=peft_model_save_dir,
        report_to = args.report_to,
        logging_dir = os.path.join(peft_model_save_dir, "logs"),
        run_name = args.run_name,
        completion_only_loss=args.completion_only_loss,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        per_device_train_batch_size = args.per_device_train_batch_size,
        per_device_eval_batch_size  = args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        eval_on_start=True,
        ddp_find_unused_parameters = args.ddp_find_unused_parameters,
        seed=args.seed,
    )
