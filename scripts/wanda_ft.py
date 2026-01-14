import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import  SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from datetime import datetime
import json
import sys
from pruning.prune import prune_wanda
from pruning.utils import check_sparsity
from SFT.utils import init_dist_if_needed, get_ranks, rank0_prune_then_sync


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
    parser.add_argument("--pruning_ratio", type=float, default=0.0, help="Wanda pruning sparsity ratio (0.0 means no pruning).")
    parser.add_argument("--pruning_nsamples", type=int, default=128, help="Number of samples for Wanda pruning calibration.")
    parser.add_argument("--pruning_seqlen", type=int, default=2048, help="Sequence length for Wanda pruning calibration.")
    parser.add_argument("--pruning_dataset_name", type=str, default="c4", help="Dataset name for Wanda pruning calibration.")
    parser.add_argument("--prune_n", type=int, default=0, help="Wanda prune_n parameter.")
    parser.add_argument("--prune_m", type=int, default=0, help="Wanda prune_m parameter.")
    parser.add_argument("--use_variant", action="store_true", help="Use Wanda variant.")

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

    is_dist = init_dist_if_needed()
    rank, local_rank, world_size = get_ranks()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,   # recommended (or float16)
        cache_dir=args.model_save_dir
    ).to(device)
    print("Model loaded.")
    print("Applying Wanda pruning...")
    if args.pruning_ratio > 0.0:
        print(f"[rank={rank}] applying Wanda (rank0-only) then syncing...")
        model = rank0_prune_then_sync(
            model,
            prune_wanda,
            sparsity_ratio=args.pruning_ratio,
            nsamples=args.pruning_nsamples,
            seed=args.seed,
            seqlen=args.pruning_seqlen,
            dataset_name=args.pruning_dataset_name,
            tokenizer=tokenizer,
            device=device,          # IMPORTANT: do not leave None
            prune_n=args.prune_n,
            prune_m=args.prune_m,
            use_variant=args.use_variant,
        )
        if rank == 0:
            print("Pruning complete and broadcasted.")
    sparsity = check_sparsity(model)
    print(f"Model sparsity after Wanda pruning: {sparsity:.2f}%")
    print("Wanda pruning applied.")
    print("Loading model for LoRA...")
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
        args=sft_config
    )
    print("Trainer prepared. Starting training...")
    trainer.train()
    print("Training completed. Saving the model...")

    merged_model = peft_model.merge_and_unload()
    merged_model = merged_model.to(device)
    print("Applying Wanda pruning to the merged model...")
    sparsity = check_sparsity(merged_model)
    print(f"Model sparsity before Wanda pruning: {sparsity:.2f}%")
    if args.pruning_ratio > 0.0:
        prune_wanda(
            sparsity_ratio=args.pruning_ratio,
            nsamples=args.pruning_nsamples,
            seed=args.seed,
            seqlen=args.pruning_seqlen,
            dataset_name=args.pruning_dataset_name,
            model=merged_model,
            tokenizer=tokenizer,
            device=device,
            prune_n=args.prune_n,
            prune_m=args.prune_m,
            use_variant=args.use_variant
        )
    sparsity = check_sparsity(merged_model)
    print(f"Model sparsity after Wanda pruning: {sparsity:.2f}%")
    print("Wanda pruning applied to the merged model.")

    merged_model.save_pretrained(peft_model_save_dir)
    print("Model saved to", peft_model_save_dir)
