import os
import torch
import argparse
from SFT.ForesightPruneCallback import ForesightPruneCallback
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import  SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from data.datasets import get_loaders
from datetime import datetime
import json
import sys


def parse_args():

    parser = argparse.ArgumentParser(description="Fine-tune a language model with LoRA and SFT.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B", help="Pre-trained model name or path.")
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
    parser.add_argument("--PBS", type=bool, default=True, help="Enable Prune-Before-Select (PBS) during training.")
    parser.add_argument("--prune_ratio", type=float, default=0, help="Pruning ratio for Foresight Pruning.")
    parser.add_argument("--mask_lr", type=float, default=0.5, help="Mask learning rate for Foresight Pruning.")
    parser.add_argument("--calib_seqlen", type=int, default=2048, help="Sequence length for calibration dataloader.")
    parser.add_argument("--calib_names", type=str, default="c4", help="Name of the calibration dataset.")
    parser.add_argument("--calib_nsamples", type=int, default=128, help="Number of samples for calibration dataloader.")
    parser.add_argument("--skip_last_n_epochs", type=int, default=0, help="Number of last epochs to skip pruning.")
    parser.add_argument("--rank0_only", type=bool, default=False, help="If True, only rank 0 performs pruning.")
    parser.add_argument("--broadcast_buffers", type=bool, default=True, help="Whether to broadcast buffers along with parameters after pruning.")


    args = parser.parse_args()

    # ---- Print args at the beginning (rank-safe for torchrun) ----
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print("\n========== RUN CONFIG ==========")
        print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
        print("Command:", " ".join(map(str, sys.argv)))
        print("Args (sorted):")
        for k, v in sorted(vars(args).items()):
            print(f"  {k}: {v}")
        print("Args (json):")
        print(json.dumps(vars(args), indent=2, sort_keys=True))
        print("================================\n")

    return args

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

    print("Loading tokenizer...")
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("Tokenizer loaded.")

    print("Preparing calibration dataloader...")
    calib_loader, _ = get_loaders(
        name=args.calib_names,
        tokenizer=tokenizer,
        seqlen=args.calib_seqlen,
        nsamples=args.calib_nsamples)
    print("Calibration dataloader prepared.")

    print("Loading the callback function used for foresight pruning...")
    cb = ForesightPruneCallback(
        dataloader=calib_loader,
        prune_ratio=args.prune_ratio,
        mask_lr=args.mask_lr,
        nsamples=args.calib_nsamples,
        merge_lora=True,
        PBS=args.PBS,
        skip_last_n_epochs=args.skip_last_n_epochs,
        rank0_only=args.rank0_only,   # recommended for DDP so only one rank does calibration/prune
        broadcast_buffers=args.broadcast_buffers,
    )
    print("Callback function loaded.")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype="auto",
        cache_dir=args.model_save_dir
    )
    peft_model = get_peft_model(model, peft_config)
    print("Model loaded.")


    print ("Loading dataset...")
    train_data = Dataset.load_from_disk(args.train_dataset_path)
    val_data = Dataset.load_from_disk(args.val_dataset_path)
    print("Dataset loaded.")

    print("Prepare Trainer...")
    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=sft_config,
        callbacks=[cb],

    )
    print("Trainer prepared. Starting training...")
    trainer.train()
    print("Training completed. Saving the model...")

    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(peft_model_save_dir)
    print("Model saved to", peft_model_save_dir)
