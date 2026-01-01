import random
import torch
import pandas as pd 
from time import time
from evaluate import load as load_metric
from tqdm import tqdm
from data.templates import *

def evaluate_HQS_rouge(model, tokenizer, 
                       file_path="data/downloaded/MEDIQA2021-Task1-TestSet-ReferenceSummaries.csv",
                       device="cuda",
                       seed=1234):
    
    random.seed(seed)
    torch.manual_seed(seed)
    """
    Summarize each question in a CSV using a simple prompt and compute ROUGE.
    """
    rouge = load_metric("rouge")
    df = pd.read_csv(file_path)
    questions = df["NLM Question"].tolist()
    reference_summaries = df["Summary"].tolist()

    model.to(device)
    model.eval()

    generated_summaries = []
    start_time = time()

    # Open a file to log question/summary pairs
    with open("generated_summaries.txt", "w", encoding="utf-8") as f_out:
        for idx, question in enumerate(tqdm(questions, desc="Generating Summaries")):
            prompt = hqs_input_template.format(input_question=question)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            ).to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=50,       # Adjust based on expected summary length
                    do_sample=True,
                    top_k=50,                  # Next-token sampling parameter
                    top_p=0.9,                 # Next-token sampling parameter
                    temperature=0.9,           # Next-token sampling parameter
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            decoded_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            # Remove the prompt from the output if it is echoed back
            if decoded_text.startswith(prompt):
                summary = decoded_text[len(prompt):].strip()
            else:
                summary = decoded_text

            generated_summaries.append(summary)

            log_entry = (
                f"\n=== Generated Summary for Question #{idx+1} ===\n"
                f"QUESTION:\n{question}\n"
                f"GENERATED SUMMARY:\n{summary}\n"
                + "=" * 50 + "\n"
            )
            f_out.write(log_entry)

    rouge_scores = rouge.compute(
        predictions=generated_summaries,
        references=reference_summaries
    )

    print("\n=== Final ROUGE SCORES ===")
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")

    total_time = time() - start_time
    print(f"\nTotal generation and evaluation time: {total_time:.2f} seconds")

    return rouge_scores, generated_summaries

