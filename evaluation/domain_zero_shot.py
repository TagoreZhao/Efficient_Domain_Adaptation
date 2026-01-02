import re
import random
import torch
import pandas as pd 
from time import time
from tqdm import tqdm
from data.templates import *
from pathlib import Path
from evaluate import load as load_metric

BAD_PREFIX_PATTERNS = [
    r"^\s*please\b.*",                 # "Please provide..."
    r"^\s*shortened summary\s*:\s*",   # "Shortened Summary: ..."
    r"^\s*input\s*:\s*",               # "Input: ..."
    r"^\s*response\s*:\s*",            # "Response: ..."
]

CUT_MARKERS = [
    "\n\n",
    "\nInput:",
    "\nShortened Summary:",
    "Shortened Summary:",
    "Input:",
    "Explanation:",
    "Please provide",
]

def evaluate_HQS_rouge(model, tokenizer, 
                       file_path="data/downloaded/MEDIQA2021-Task1-TestSet-ReferenceSummaries.xlsx",\
                       save_output="assets/HQS_generated_summaries.txt",
                       max_new_tokens=50,
                       device="cuda",
                       seed=1234):
    
    random.seed(seed)
    torch.manual_seed(seed)
    """
    Summarize each question in an Excel file using a simple prompt and compute ROUGE.
    """
    rouge = load_metric("rouge")
    df = pd.read_excel(file_path)
    questions = df["NLM Question"].tolist()
    reference_summaries = df["Summary"].tolist()

    model.to(device)
    model.eval()

    generated_summaries = []
    start_time = time()
    
    save_path = Path(save_output)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Open a file to log question/summary pairs
    with open(save_path, "w", encoding="utf-8") as f_out:
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
                    max_new_tokens=max_new_tokens,       # Adjust based on expected summary length
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

            summary = clean_first_sentence(summary)
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

def clean_first_sentence(text: str, *, max_words: int = 40, min_chars: int = 15) -> str:
    """
    Return a short, evaluation-friendly summary:
    - drop boilerplate/instructions
    - keep only the first valid sentence/line/word-chunk
    """
    if not text:
        return ""

    t = text.strip()

    # Hard cut at known markers that often begin rambling/meta output
    for m in CUT_MARKERS:
        idx = t.find(m)
        if idx != -1 and idx > 0:
            t = t[:idx].strip()
            break

    # Normalize whitespace
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\r\n?", "\n", t).strip()

    # Split into candidate chunks: first try sentence-like split
    # If no punctuation, fallback to newline split, then to word truncation.
    sentence_candidates = re.split(r"(?<=[.!?])\s+", t) if re.search(r"[.!?]", t) else [t]
    candidates = []
    for s in sentence_candidates:
        s = s.strip()
        if not s:
            continue
        # If there are newlines, take only the first line for that chunk
        s = s.split("\n", 1)[0].strip()
        candidates.append(s)

    # Filter out instruction-like first sentences
    def looks_like_boilerplate(s: str) -> bool:
        low = s.lower().strip()
        for pat in BAD_PREFIX_PATTERNS:
            if re.match(pat, low):
                return True
        return False

    # Choose first candidate that isn't boilerplate and is long enough
    chosen = ""
    for c in candidates:
        if looks_like_boilerplate(c):
            continue
        if len(c) >= min_chars:
            chosen = c
            break

    if not chosen:
        # fallback: take first non-empty line
        first_line = t.split("\n", 1)[0].strip()
        chosen = "" if looks_like_boilerplate(first_line) else first_line

    # Final fallback: truncate words if still empty or too short
    if not chosen or len(chosen) < min_chars:
        words = t.replace("\n", " ").split()
        chosen = " ".join(words[:max_words]).strip()

    # Enforce max_words
    words = chosen.split()
    if len(words) > max_words:
        chosen = " ".join(words[:max_words]).strip()

    return chosen