import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.domain_zero_shot import *
from evaluation.perplexity import eval_ppl
from trl import  SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model
from data.datasets import construct_med_data, construct_legal_data
from data.utils import to_prompt_completion_hf
from SFT.utils import save_run_manifest

torch.cuda.empty_cache()
os.environ["WANDB_PROJECT"] = "Efficient_Domain_Adaptation"
os.environ["WANDB_DIR"] = "./assets/wandb"
model_name = "Qwen/Qwen3-1.7B"
model_save_dir = "model/downloaded"
run_name = "qwen3_1.7b_med_3eps_all_token_batch16"
peft_model_save_dir = os.path.join(model_save_dir, run_name)
seed = 42

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
    completion_only_loss=False,
    do_eval=True,
    eval_strategy="steps",
    eval_steps=100,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size  = 4,
    learning_rate=1e-4,
    weight_decay=0.01,
    num_train_epochs=3,
    eval_on_start=True,
    ddp_find_unused_parameters = False,
    seed=seed,
)

# dataset_path = 'data/downloaded/test_medical_data'

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print("Tokenizer loaded.")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    cache_dir=model_save_dir
)
peft_model = get_peft_model(model, peft_config)
print("Model loaded.")

print ("Preparing dataset...")
data_train = construct_med_data(pubmed_count = 7000,
                                hqs_count= 1000,
                                mednli_count= 7000,
                                split='train',
                                seed=seed)
print(f"Number of training examples: {len(data_train)}")

data_validation = construct_med_data(pubmed_count = 500,
                                    hqs_count= 100,
                                    mednli_count= 1000,
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
    args=sft_config
)
print("Trainer prepared. Starting training...")
trainer.train()
print("Training completed. Saving the model...")

merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(peft_model_save_dir)
print("Model saved to", peft_model_save_dir)

print("Evaluating model on PubMedQA...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="cuda",
    cache_dir=model_save_dir
)

pre = {}
acc, maf, cm, predictions = evaluate_pubmedqa(model, tokenizer)
print(f"PubMedQA Accuracy before fine-tuning: {acc:.4f}, Macro F1: {maf:.4f}")
pre["PubMedQA"] = {"accuracy": float(acc), "macro_f1": float(maf)}
acc, maf, cm, predictions = evaluate_mednli(model, tokenizer)
print(f"MedNLI Accuracy before fine-tuning: {acc:.4f}, Macro F1: {maf:.4f}")
pre["MedNLI"] = {"accuracy": float(acc), "macro_f1": float(maf)}
rouge, prediction = evaluate_hqs(model, tokenizer)
print("HQS ROUGE before fine-tuning:", rouge)
pre["HQS"] = {"rouge": float(rouge)}


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


ppl = eval_ppl(model=model, tokenizer=tokenizer, dataset='harrison', seqlen=2048, device=torch.device("cuda:0"), seed=seed)
print(f"Perplexity on Harrison before fine-tuning: {ppl:.4f}")
ppl = eval_ppl(model=merged_model, tokenizer=tokenizer, dataset='harrison', seqlen=2048, device=torch.device("cuda:0"), seed=seed)
print(f"Perplexity on Harrison after fine-tuning: {ppl:.4f}")


# manifest_path = save_run_manifest(
#     out_dir=peft_model_save_dir,
#     model_name=model_name,
#     peft_config=peft_config,
#     sft_config=sft_config,
#     tokenizer_name=model_name,
#     train_len=len(train_dataset),
#     val_len=len(val_dataset),
#     pre_metrics=pre,
#     post_metrics=post,
#     model_for_count=peft_model,  # counts trainable LoRA params
# )
# print("Saved manifest to:", manifest_path)
