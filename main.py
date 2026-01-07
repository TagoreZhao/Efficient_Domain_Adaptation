import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.domain_zero_shot import *
import torch
from datasets import load_from_disk
from trl import  SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model
from data.datasets import construct_med_data, get_pubmedqa

# data_train = get_pubmedqa(pubmed_count = 10, split='train')
# print(f"Number of training examples: {len(data_train)}")
# print("Sample training example:")
# print(data_train[0])
# print("\n")
# data_validation = get_pubmedqa(pubmed_count = 10, split='validation')
# print(f"Number of validation examples: {len(data_validation)}")
# print("Sample validation example:")
# print(data_validation[0])

data_train = construct_med_data(pubmed_count=3,
                                mednli_count=3,
                                hqs_count=3,
                                seed = 123,
                                split='train')
print(f"Number of training examples: {len(data_train)}")
print("Sample training example:")
print(data_train[0])
print("\n")
data_validation = construct_med_data(pubmed_count=3,
                                     mednli_count=3, 
                                     hqs_count=3,
                                     seed = 123,
                                     split='validation')
print(f"Number of validation examples: {len(data_validation)}")
print("Sample validation example:")
print(data_validation[0])
