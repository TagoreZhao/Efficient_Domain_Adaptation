import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from evaluation.perplexity import eval_ppl
from pruning.utils import check_sparsity
from pruning.prune import foresight_prune
torch.cuda.empty_cache()

# model_name = "meta-llama/Llama-3.2-1B"
model_name = "Qwen/Qwen3-0.6B"
model_save_dir = "model/downloaded"
peft_config = LoraConfig(
    r = 8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none" ,
    target_modules=["q_proj","v_proj", "o_proj", "k_proj", "up_proj", "down_proj", "gate_proj"],
)

os.makedirs(model_save_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="cuda",
    cache_dir=model_save_dir
)
peft_model = get_peft_model(model, peft_config)

ppl = eval_ppl(model = model, tokenizer=tokenizer, dataset="wikitext2", seqlen=2048)
print (f"Perplexity before pruning: {ppl}")

foresight_prune(peft_model,
                prune_ratio=0.3,
                mask_lr=0.5,
                nsamples=128,
                seed=42,
                tokenizer=tokenizer)

ppl = eval_ppl(model = peft_model, tokenizer=tokenizer, dataset="wikitext2", seqlen=2048)
print (f"Perplexity before pruning: {ppl}")