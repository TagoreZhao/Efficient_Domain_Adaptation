import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.domain_zero_shot import *
from trl import  SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model
from data.datasets import construct_med_data, get_loaders
from data.utils import to_prompt_completion_hf
from SFT.ForesightPruneCallback import ForesightPruneCallback
from pruning.utils import check_sparsity

torch.cuda.empty_cache()
os.environ["WANDB_PROJECT"] = "test"
os.environ["WANDB_DIR"] = "./assets/wandb"
model_name = "Qwen/Qwen3-0.6B"
model_save_dir = "model/downloaded"
peft_model_save_dir = os.path.join(model_save_dir, "qwen3_0.6b_med_3eps_foresight0.5")
# merged_model_save_dir = os.path.join(model_save_dir, "merged_qwen3_0.6b_med_3eps_test")
seed = 42
prune_ratio = 0.4
mask_lr = 0.5
run_name = "qwen3_0.6b_med_3eps_foresight0.5"

peft_config = LoraConfig(
    r = 8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none" ,
    target_modules=["q_proj","v_proj", "o_proj", "k_proj", "up_proj", "down_proj", "gate_proj"],
)

sft_config = SFTConfig(
    output_dir=peft_model_save_dir,
    report_to = "wandb",
    logging_dir = os.path.join(peft_model_save_dir, "logs"),
    run_name = run_name,
    completion_only_loss=True,
    do_eval=True,
    eval_strategy="steps",
    eval_steps=50,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 16,
    learning_rate=2e-4,
    weight_decay=0.01,
    num_train_epochs=3,
    eval_on_start=True,
)

# dataset_path = 'data/downloaded/test_medical_data'

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print("Tokenizer loaded.")

print("Preparing calibration dataloader...")
calib_loader, _ = get_loaders(
    name="harrison",
    tokenizer=tokenizer,
    seqlen=2048,
    nsamples=128)
print("Calibration dataloader prepared.")

print("Loading the callback function used for foresight pruning...")
cb = ForesightPruneCallback(
    dataloader=calib_loader,
    prune_ratio=prune_ratio,
    mask_lr=mask_lr,
    nsamples=128,
    merge_lora=True,
    PBS=True,
    skip_last_n_epochs=0,
    rank0_only=True,   # recommended for DDP so only one rank does calibration/prune
)
print("Callback function loaded.")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    cache_dir=model_save_dir
)
peft_model = get_peft_model(model, peft_config)
print("Model loaded.")

print ("Preparing dataset...")
data_train = construct_med_data(pubmed_count = 70,
                                hqs_count= 10,
                                mednli_count= 70,
                                split='train',
                                seed=seed)
print(f"Number of training examples: {len(data_train)}")

data_validation = construct_med_data(pubmed_count = 100,
                                    hqs_count= 100,
                                    mednli_count= 100,
                                    split='validation',
                                    seed=seed)
print(f"Number of validation examples: {len(data_validation)}")

train_dataset = to_prompt_completion_hf(data_train, tokenizer=tokenizer)
val_dataset = to_prompt_completion_hf(data_validation, tokenizer=tokenizer)
train_dataset.save_to_disk(os.path.join(peft_model_save_dir, "med_train_dataset"))
val_dataset.save_to_disk(os.path.join(peft_model_save_dir, "med_val_dataset"))
print("Datasets converted to HuggingFace format. Dataset preparation completed.")


print("prepare trainer...")
trainer = SFTTrainer(
    model=peft_model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=sft_config,
    callbacks=[cb],
)
print("Trainer prepared. Starting training...")
trainer.train()
print("Training completed. Saving the model...")

peft_model.save_pretrained(peft_model_save_dir)
tokenizer.save_pretrained(peft_model_save_dir)
print("PEFT Model saved to", peft_model_save_dir)

merged_model = peft_model.merge_and_unload()
sparsity = check_sparsity(merged_model)
print(f"Merged model sparsity: {sparsity:.2f}%")
# merged_model.save_pretrained(merged_model_save_dir)
# print("Model saved to", merged_model_save_dir)

# print("Evaluating model on PubMedQA...")
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     dtype="auto",
#     device_map="cuda",
#     cache_dir=model_save_dir
# )

# pre = {}
# acc, maf, cm, predictions = evaluate_pubmedqa(model, tokenizer)
# print(f"PubMedQA Accuracy before fine-tuning: {acc:.4f}, Macro F1: {maf:.4f}")
# pre["PubMedQA"] = {"accuracy": float(acc), "macro_f1": float(maf)}
# acc, maf, cm, predictions = evaluate_mednli(model, tokenizer)
# print(f"MedNLI Accuracy before fine-tuning: {acc:.4f}, Macro F1: {maf:.4f}")
# pre["MedNLI"] = {"accuracy": float(acc), "macro_f1": float(maf)}
# rouge, prediction = evaluate_hqs(model, tokenizer)
# print("HQS ROUGE before fine-tuning:", rouge)
# pre["HQS"] = {"rouge": rouge}


post = {}
acc, maf, cm, predictions = evaluate_pubmedqa(merged_model, tokenizer)
print(f"PubMedQA Accuracy after fine-tuning: {acc:.4f}, Macro F1: {maf:.4f}")
post["PubMedQA"] = {"accuracy": float(acc), "macro_f1": float(maf)}
acc, maf, cm, predictions = evaluate_mednli(merged_model, tokenizer)
print(f"MedNLI Accuracy after fine-tuning: {acc:.4f}, Macro F1: {maf:.4f}")
post["MedNLI"] = {"accuracy": float(acc), "macro_f1": float(maf)}
rouge, prediction = evaluate_hqs(merged_model, tokenizer)
print("HQS ROUGE after fine-tuning:", rouge)
post["HQS"] = {"rouge": rouge}
