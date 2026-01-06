
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.domain_zero_shot import *
import torch
from datasets import load_from_disk
from trl import  SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model

torch.cuda.empty_cache()

model_name = "Qwen/Qwen3-0.6B"
model_save_dir = "model/downloaded"
peft_model_save_dir = os.path.join(model_save_dir, "test_qwen3_medical_lora")
dataset_path = 'data/downloaded/test_medical_data'
# os.makedirs(model_save_dir, exist_ok=True)

# # load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

peft_config = LoraConfig(
    r= 8,
    lora_alpha=16,
    lora_dropout=0,
    bias="none" ,
    target_modules=["q_proj","v_proj"],
)

sft_config = SFTConfig(
    output_dir=peft_model_save_dir,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="cuda",
    cache_dir=model_save_dir
)

peft_model = get_peft_model(model, peft_config)

peft_model.to(device=torch.device("cuda"))
dataset = load_from_disk(dataset_path)

print("Dataset loaded. Starting training...")
trainer = SFTTrainer(
    model=peft_model,
    train_dataset=dataset,
    args=sft_config
)

trainer.train()
print("Training completed. Saving the model...")
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(peft_model_save_dir)


print("Model saved to", peft_model_save_dir)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="cuda",
    cache_dir=model_save_dir
)
acc, maf, cm, predictions = evaluate_pubmedqa(model, tokenizer)
print(f"PubMedQA Accuracy before fine-tuning: {acc:.4f}, Macro F1: {maf:.4f}")
acc, maf, cm, predictions = evaluate_pubmedqa(merged_model, tokenizer)
print(f"PubMedQA Accuracy after fine-tuning: {acc:.4f}, Macro F1: {maf:.4f}") 