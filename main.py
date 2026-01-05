import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.domain_zero_shot import evaluate_mednli, evaluate_hqs, evaluate_pubmedqa
import torch
import pandas as pd
from data.templates import *
from tqdm import tqdm

torch.cuda.empty_cache()

# model_name = "meta-llama/Llama-3.2-1B"
# model_save_dir = "model/downloaded"
# os.makedirs(model_save_dir, exist_ok=True)

# thinking = True
# # load the tokenizer and the model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     dtype="auto",
#     device_map="auto",
#     cache_dir=model_save_dir
# )

from data.datasets import get_casehold, get_billsum, get_contractnli, construct_legal_data
legal_data = construct_legal_data()
print (f"Total legal data instances: {len(legal_data)}")
for i in range(5):
    print("Instruction:", legal_data[i]["source"])
    print("Input:", legal_data[i]["input_text"])
    print("Output:", legal_data[i]['target_text'])
    print("-----")
# import datasets
# print(datasets.__version__)

# rouge_scores, summaries = evaluate_hqs(model, tokenizer, enable_thinking=False)
# print(f"HQS ROUGE Scores: {rouge_scores}")
# acc, maf, cm, predictions = evaluate_pubmedqa(model, tokenizer)
# print(f"PubMedQA Accuracy: {acc}, Macro F1: {maf}")
# acc, maf, cm, predictions = evaluate_mednli(model, tokenizer, n_eval=50)
# print(f"MedNLI Accuracy: {acc}, Macro F1: {maf}")