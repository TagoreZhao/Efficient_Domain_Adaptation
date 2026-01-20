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
start_id = 11
end_id = 30  # inclusive
seed = 1234
calib_dataset = "c4"  # "harrison" or "c4"
os.makedirs(model_save_dir, exist_ok=True)
csv_path = os.path.join(out_dir, f"foresight_{calib_dataset}_metrics.csv")

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
    name=calib_dataset,
    tokenizer=tokenizer,
    seqlen=2048,
    nsamples=128)

foresight_prune(model=peft_model,
                dataloader=dataloader,
                prune_ratio=0.3,
                mask_lr=0.5,
                nsamples=128,
                PBS = False)

# model = peft_model.merge_and_unload()
# prune_wanda(sparsity_ratio=0.3,
#             nsamples=128,
#             seed=42,
#             seqlen=2048,
#             model=model,
#             tokenizer=tokenizer,
#             dataset_name=calib_dataset)

model = peft_model.merge_and_unload()
sparsity = check_sparsity(model)
print(f"Sparsity after pruning: {sparsity:.2%}")

ppl = eval_ppl(model = model, tokenizer=tokenizer, dataset="harrison", seqlen=2048)
print (f"Perplexity after pruning: {ppl}")

rows = []

for run_id in range(start_id, end_id + 1):
    print(f"=== Run ID: {run_id} ===")
    run_seed = seed + (run_id - 1)

    pub_acc, pub_maf, pub_cm, pub_preds = evaluate_pubmedqa(model, tokenizer, seed=run_seed)
    med_acc, med_maf, med_cm, med_preds = evaluate_mednli(model, tokenizer, seed=run_seed)
    rouge, hqs_preds = evaluate_hqs(model, tokenizer, seed=run_seed)

    row = {
        "run_id": run_id,
        "seed": run_seed,
        "calib_dataset": calib_dataset,
        "sparsity": float(sparsity),
        "harrison ppl": float(ppl),
        "pubmedqa_acc": float(pub_acc),
        "pubmedqa_macro_f1": float(pub_maf),
        "mednli_acc": float(med_acc),
        "mednli_macro_f1": float(med_maf),
        "hqs_rouge1": float(rouge["rouge1"]),
        "hqs_rouge2": float(rouge["rouge2"]),
        "hqs_rougeL": float(rouge["rougeL"]),
        "pubmedqa_cm_json": json.dumps(pub_cm.tolist() if hasattr(pub_cm, "tolist") else pub_cm),
        "mednli_cm_json": json.dumps(med_cm.tolist() if hasattr(med_cm, "tolist") else med_cm),
    }

    # Write this iteration immediately
    df_row = pd.DataFrame([row])
    file_exists = os.path.exists(csv_path)
    df_row.to_csv(csv_path, mode="a", header=not file_exists, index=False)

    print(f"Appended row to: {csv_path}")
