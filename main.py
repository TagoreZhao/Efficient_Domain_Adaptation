from ast import List
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.domain_zero_shot import *
import torch
import pandas as pd
from data.templates import *
from tqdm import tqdm
from data.datasets import construct_med_data
from data.utils import *
torch.cuda.empty_cache()

# model_name = "Qwen/Qwen3-0.6B"
# model_save_dir = "model/downloaded"
# os.makedirs(model_save_dir, exist_ok=True)

# # load the tokenizer and the model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     dtype="auto",
#     device_map="auto",
#     cache_dir=model_save_dir
# )


medical_finetune_data = construct_med_data(seed = 1354)


medical_finetune_data_prompt_completion = to_prompt_completion(
    medical_finetune_data,
    input_key="input_text",
    target_key="target_text",)
print(medical_finetune_data_prompt_completion[0])


medical_finetune_data_hf = to_prompt_completion_hf(
    medical_finetune_data,
    features=default_prompt_completion_features,
    input_key="input_text",
    target_key="target_text",
)

print(medical_finetune_data_hf[0])
print(medical_finetune_data_hf)