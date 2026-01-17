import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from evaluation.perplexity import eval_ppl
from pruning.utils import check_sparsity
from pruning.prune import foresight_prune, prune_wanda
from evaluation.domain_zero_shot import *
import pandas as pd
from data.datasets import get_loaders

torch.cuda.empty_cache()

# model_name = "meta-llama/Llama-3.2-1B"

model_name = "Qwen/Qwen3-0.6B"
model_save_dir = "model/downloaded"
adapter_save_dir = "model/downloaded/qwen3_0.6b_med_3eps_test"
out_dir = "assets/pruning_temp_results"
eval_reps = 10
seed = 1234
os.makedirs(model_save_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="cuda",
    cache_dir=model_save_dir
)
# peft_model = get_peft_model(model, peft_config)
peft_model = PeftModel.from_pretrained(
    model,
    adapter_save_dir)

dataloader, _ = get_loaders(
    name="harrison",
    tokenizer=tokenizer,
    seqlen=2048,
    nsamples=128)

foresight_prune(model=peft_model,
                dataloader=dataloader,
                prune_ratio=0.4,
                mask_lr=0.5,
                nsamples=128,
                PBS = True)

# model = peft_model.merge_and_unload()
# prune_wanda(sparsity_ratio=0.4,
#             nsamples=128,
#             seed=42,
#             seqlen=2048,
#             model=model,
#             tokenizer=tokenizer,
#             dataset_name="harrison")

model = peft_model.merge_and_unload()
sparsity = check_sparsity(model)
print(f"Sparsity after pruning: {sparsity:.2%}")

ppl = eval_ppl(model = model, tokenizer=tokenizer, dataset="harrison", seqlen=2048)
print (f"Perplexity after pruning: {ppl}")

rows = []

for i in range(eval_reps):
    run_seed = seed + i
    pub_acc, pub_maf, pub_cm, pub_preds = evaluate_pubmedqa(model, tokenizer, seed=run_seed)
    med_acc, med_maf, med_cm, med_preds = evaluate_mednli(model, tokenizer, seed=run_seed)
    rouge, hqs_preds = evaluate_hqs(model, tokenizer, seed=run_seed)
    row = {
        "run_id": i,
        "seed": run_seed,
        "pubmedqa_acc": float(pub_acc),
        "pubmedqa_macro_f1": float(pub_maf),
        "mednli_acc": float(med_acc),
        "mednli_macro_f1": float(med_maf),
        "hqs_rouge1": float(rouge["rouge1"]),
        "hqs_rouge2": float(rouge["rouge2"]),
        "hqs_rougeL": float(rouge["rougeL"]),
    }

    # Optional: store confusion matrices as JSON strings (so CSV stays valid)
    # If pub_cm / med_cm are numpy arrays or torch tensors, convert to lists first.
    row["pubmedqa_cm_json"] = json.dumps(pub_cm.tolist() if hasattr(pub_cm, "tolist") else pub_cm)
    row["mednli_cm_json"]   = json.dumps(med_cm.tolist() if hasattr(med_cm, "tolist") else med_cm)

    rows.append(row)

df = pd.DataFrame(rows)
os.makedirs(out_dir, exist_ok=True)

csv_path = os.path.join(out_dir, "foresightPBS40_metrics.csv")
df.to_csv(csv_path, index=False)

print("Saved:", csv_path)