import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.domain_zero_shot import evaluate_mednli, evaluate_hqs, evaluate_pubmedqa
import torch
import pandas as pd
from data.templates import *
from tqdm import tqdm

torch.cuda.empty_cache()

model_name = "meta-llama/Llama-3.2-1B"
model_save_dir = "model/downloaded"
os.makedirs(model_save_dir, exist_ok=True)

thinking = True
# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    cache_dir=model_save_dir
)


# device = "cuda" if torch.cuda.is_available() else "cpu"

# file_path="data/downloaded/MEDIQA2021-Task1-TestSet-ReferenceSummaries.xlsx"
# df = pd.read_excel(file_path)
# questions = df["NLM Question"].tolist()
# reference_summaries = df["Summary"].tolist()
# for idx, question in enumerate(tqdm(questions, desc="Generating Summaries")):
#     if thinking:
#         prompt = hqs_input_template.format(input_question=question)
#         prompt =[
#             {"role": "user", "content": prompt}
#         ]
#         prompt = tokenizer.apply_chat_template(
#                     prompt,
#                     tokenize=False,
#                     add_generation_prompt=True,
#                     enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
#                 )
#         inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
#     else:
#         prompt = hqs_input_template.format(input_question=question)
#         inputs = tokenizer(
#             prompt,
#             return_tensors="pt",
#             max_length=1024,
#             truncation=True
#         ).to(device)

#     with torch.no_grad():
#         output_ids = model.generate(
#             **inputs,
#             max_new_tokens=1000,       # Adjust based on expected summary length
#             do_sample=True,
#             top_k=50,                  # Next-token sampling parameter
#             top_p=0.9,                 # Next-token sampling parameter
#             temperature=0.9,           # Next-token sampling parameter
#             pad_token_id=tokenizer.pad_token_id,
#             eos_token_id=tokenizer.eos_token_id
#         )
#     if thinking:
#         output_ids = output_ids[0][len(inputs.input_ids[0]):].tolist() 
#         try:
#             # rindex finding 151668 (</think>)
#             index = len(output_ids) - output_ids[::-1].index(151668)
#         except ValueError:
#             index = 0
#         thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
#         content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

#         print("thinking content:", thinking_content)
#         print("content:", content)
#     else:
#         decoded_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
#         # Remove the prompt from the output if it is echoed back
#         if decoded_text.startswith(prompt):
#             summary = decoded_text[len(prompt):].strip()
#         else:
#             summary = decoded_text
#         print(f"Question {idx+1}: {question}")
#         print(f"Generated Summary: {summary}")

#     if idx >= 1:
#         break

rouge_scores, summaries = evaluate_hqs(model, tokenizer, enable_thinking=False)
print(f"HQS ROUGE Scores: {rouge_scores}")
# acc, maf, cm, predictions = evaluate_pubmedqa(model, tokenizer)
# print(f"PubMedQA Accuracy: {acc}, Macro F1: {maf}")
# acc, maf, cm, predictions = evaluate_mednli(model, tokenizer, n_eval=50)
# print(f"MedNLI Accuracy: {acc}, Macro F1: {maf}")