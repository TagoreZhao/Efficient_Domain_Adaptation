import re
import json
import random
import torch
import pandas as pd 
from time import time
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from collections import Counter
from data.templates import *
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_recall_fscore_support
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

def evaluate_contractnli(
    model,
    tokenizer,
    device="cuda",
    save_output="assets/contractnli_predictions.txt",
    seed=1234,
    max_attempts=3,
    max_new_tokens=10,
    n_eval=200,              # optional: limit number of examples
    do_bootstrap_ci=True,
    n_boot=1000,
):
    random.seed(seed)
    torch.manual_seed(seed)
    model.to(device)
    tokenizer.pad_token = tokenizer.eos_token
    data = load_dataset("kiddothe2b/contract-nli", "contractnli_a", split="test")
    expected_labels = ["entailment", "contradiction", "neutral"]
    labels = data["label"][:n_eval] if n_eval is not None else data["label"]
    premises = data["premise"][:n_eval] if n_eval is not None else data["premise"]
    hypotheses = data["hypothesis"][:n_eval] if n_eval is not None else data["hypothesis"]
    predictions, ground_truth = [], []
    
    with open(save_output, "w", encoding="utf-8") as f_out:
        f_out.write("=== ContractNLI Evaluation Log ===\n")
        f_out.write(
            f"n_eval={len(data) if n_eval is None else n_eval}, "
            f"seed={seed}, max_new_tokens={max_new_tokens}, max_attempts={max_attempts}\n"
        )
        f_out.write(f"device={device}\n")
        f_out.write("=" * 70 + "\n\n")

        for idx, premise in enumerate(tqdm(premises, desc="Evaluating ContractNLI")):
            hypothesis = hypotheses[idx]
            gold_label = {1: "entailment", 2: "neutral", 0: "contradiction"}.get(labels[idx], "unknown")
            ground_truth.append(gold_label)
            prompt = mednli_input_template.format(sentence1=premise, sentence2=hypothesis)
            prediction = "unknown"
            for attempt in range(max_attempts):
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=2048,
                ).to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        top_k=50,
                        top_p=0.9,
                        temperature=0.9,
                        output_scores=True,
                        return_dict_in_generate=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                outputs = outputs.sequences[0][len(inputs.input_ids[0])-10:].tolist()
                decoded_output = tokenizer.decode(outputs, skip_special_tokens=True).lower()
                match = re.search(
                            r"\bTheir relationship is\b\s*[:\-]?\s*"
                            r"(?:\*\*|\*)?\s*"          # optional markdown emphasis opening
                            r"(?:['\"]{1,2})?\s*"       # optional quote wrapper: ' or " or '' or ""
                            r"(entailment|contradiction|neutral)"
                            r"\s*(?:['\"]{1,2})?\s*"    # optional closing quotes
                            r"(?:\*\*|\*)?\b",          # optional markdown emphasis closing
                            decoded_output,
                            flags=re.IGNORECASE
                        )
                if match:
                    prediction = match.group(1).strip()
                    break

            predictions.append(prediction)

            f_out.write(f"ID: {idx}\n")
            f_out.write(f"GOLD: {gold_label}\n")
            f_out.write(f"PRED: {prediction}\n")
            f_out.write(f"PREMISE:\n")
            f_out.write(f"{premise.strip()}\n")
            f_out.write(f"HYPOTHESIS:\n")
            f_out.write(f"{hypothesis.strip()}\n")
            f_out.write(f"RAW_DECODED:\n{decoded_output.strip()}\n")
            f_out.write("-" * 70 + "\n\n")

        accuracy = accuracy_score(ground_truth, predictions)
        macro_f1 = f1_score(ground_truth, predictions, average="macro", labels=expected_labels)
        cm = confusion_matrix(ground_truth, predictions, labels=expected_labels)
        prec, rec, f1s, support = precision_recall_fscore_support(
            ground_truth, predictions, labels=expected_labels, zero_division=0
        )
        pred_counts = Counter(predictions)
        true_counts = Counter(ground_truth)
        report = classification_report(ground_truth, predictions, labels=expected_labels, zero_division=0)
        ci_text = ""
        if do_bootstrap_ci:
            rng = random.Random(seed)
            n = len(ground_truth)
            boot_scores = []
            for _ in range(n_boot):
                idxs = [rng.randrange(n) for _ in range(n)]
                t_b = [ground_truth[i] for i in idxs]
                p_b = [predictions[i] for i in idxs]
                boot_scores.append(f1_score(t_b, p_b, average="macro", labels=expected_labels))
            boot_scores.sort()
            lo = boot_scores[int(0.025 * n_boot)]
            hi = boot_scores[int(0.975 * n_boot) - 1]
            ci_text = f"Macro-F1 95% bootstrap CI (n_boot={n_boot}): [{lo:.4f}, {hi:.4f}]"
        
        f_out.write("\n\n" + "=" * 70 + "\n")
        f_out.write("=== FINAL METRICS SUMMARY ===\n")
        f_out.write(f"Accuracy: {accuracy:.4f}\n")
        f_out.write(f"Macro-F1: {macro_f1:.4f}\n")
        if ci_text:
            f_out.write(ci_text + "\n")

        f_out.write("\n=== Prediction Distribution ===\n")
        for label in expected_labels + sorted([l for l in pred_counts.keys() if l not in expected_labels]):
            if label in pred_counts:
                count = pred_counts[label]
                f_out.write(f"  {label}: {count} ({count/len(predictions)*100:.2f}%)\n")

        f_out.write("\n=== Ground Truth Distribution ===\n")
        for label in expected_labels + sorted([l for l in true_counts.keys() if l not in expected_labels]):
            if label in true_counts:
                count = true_counts[label]
                f_out.write(f"  {label}: {count} ({count/len(ground_truth)*100:.2f}%)\n")

        f_out.write("\n=== Confusion Matrix (rows=true, cols=pred) ===\n")
        f_out.write("Labels: " + ", ".join(expected_labels) + "\n")
        f_out.write(str(cm) + "\n")

        f_out.write("\n=== Per-class Precision/Recall/F1/Support ===\n")
        for i, lbl in enumerate(expected_labels):
            f_out.write(
                f"  {lbl}: precision={prec[i]:.4f}, recall={rec[i]:.4f}, f1={f1s[i]:.4f}, support={support[i]}\n"
            )

        f_out.write("\n=== Classification Report ===\n")
        f_out.write(report + "\n")

        f_out.write("\n=== Error Rates per True Label ===\n")
        for i, lbl in enumerate(expected_labels):
            total = cm[i].sum()
            correct = cm[i][i]
            err = 1.0 - (correct / total) if total > 0 else 0.0
            f_out.write(f"  {lbl}: {err*100:.2f}%\n")

    print("=== ContractNLI Results ===")
    print(f"Saved detailed log to: {save_output}")
    print(f"Accuracy = {accuracy:.4f}")
    print(f"Macro-F1 = {macro_f1:.4f}")
    if ci_text:
        print(ci_text)

    return accuracy, macro_f1, cm, predictions


def evaluate_casehold(model, 
                      tokenizer, 
                      device="cuda", 
                      save_output="assets/casehold_predictions.txt",
                      seed=1234,
                      max_attempts=3,
                      max_new_tokens=10,
                      n_eval=200, 
                      do_bootstrap_ci=True,
                      n_boot=1000):
    """
    Evaluate the model on the first `n_eval` of the CaseHold dataset.
    First tries regex extraction; if fails due to invalid probabilities, falls back to greedy decoding.
    Logs when fallback occurs and inspects for NaN/Inf in logits.
    Prints full evaluation metrics, distributions, confusion matrix, and classification report.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    save_path = Path(save_output)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_dataset("casehold/casehold", split="test", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    predictions = {}
    citing_prompt = {row['example_id']: str(row["citing_prompt"]).strip().lower() for row in data}
    holdings = {row['example_id']: [row["holding_0"], row["holding_1"], row["holding_2"], row["holding_3"], row["holding_4"]] for row in data}
    example_ids = list(citing_prompt.keys())[:n_eval]
    ground_truth = ground_truth = {row["example_id"]: str(row["label"]).strip().lower() for row in data}

    expected_labels = [str(i) for i in range(5)]

    with open(save_path, "w", encoding="utf-8") as f_out:
        f_out.write("=== CaseHold Evaluation Log ===\n")
        f_out.write(f"n_eval={n_eval}, seed={seed}, max_new_tokens={max_new_tokens}\n")
        f_out.write(f"device={device}\n")
        f_out.write("=" * 70 + "\n\n")

        for example_id in tqdm(example_ids, desc="Evaluating CaseHold"):
            prompt = casehold_input_template.format(citing_prompt=citing_prompt[example_id],
                                                    holding_0=holdings[example_id][0],
                                                    holding_1=holdings[example_id][1],
                                                    holding_2=holdings[example_id][2],
                                                    holding_3=holdings[example_id][3],
                                                    holding_4=holdings[example_id][4])
            prediction = "unknown"

            for attempt in range(max_attempts):
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=3500
                ).to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=True,
                        temperature=0.9,
                        top_p=0.9,
                        output_scores=True,
                        return_dict_in_generate=True
                    )

                outputs = outputs.sequences[0][len(inputs.input_ids[0])-10:].tolist()
                decoded_output = tokenizer.decode(outputs, skip_special_tokens=True).lower()
                match = re.search(r"the answer is\s*([0-4])\b", decoded_output)
                if match:
                    prediction = int(match.group(1))
            
            gold = ground_truth.get(example_id, "unknown")
            predictions[example_id] = prediction

            f_out.write(f"example_id: {example_id}\n")
            f_out.write(f"GOLD: {gold}\n")
            f_out.write(f"PRED: {prediction}\n")
            f_out.write("CITING PROMPT:\n")
            f_out.write(citing_prompt[example_id].strip() + "\n")
            f_out.write("HOLDINGS:\n")
            for idx, holding in enumerate(holdings[example_id]):
                f_out.write(f"  {idx}: {holding.strip()}\n")
            f_out.write("RAW_DECODED:\n")
            f_out.write(decoded_output.strip() + "\n")
            f_out.write("-" * 70 + "\n\n")
        
        truth = [str(ground_truth[eid]) for eid in example_ids]
        preds = [str(predictions[eid]) for eid in example_ids]

        print("length of truth:", len(truth))
        print("length of preds:", len(preds))

        # ==== Metrics ====
        acc = accuracy_score(truth, preds)
        maf = f1_score(truth, preds, average="macro")

        cm = confusion_matrix(truth, preds, labels=expected_labels)

        prec, rec, f1s, support = precision_recall_fscore_support(truth, preds, labels=expected_labels, zero_division=0)

        pred_counts = Counter(preds)
        true_counts = Counter(truth)

        report = classification_report(
                truth, preds, labels=expected_labels, zero_division=0
            )
        # Optional bootstrap CI for macro-F1
        ci_text = ""
        if do_bootstrap_ci:
            rng = random.Random(seed)
            n = len(truth)
            boot_scores = []
            for _ in range(n_boot):
                idxs = [rng.randrange(n) for _ in range(n)]
                t_b = [truth[i] for i in idxs]
                p_b = [preds[i] for i in idxs]
                boot_scores.append(f1_score(t_b, p_b, average="macro", labels=expected_labels))
            boot_scores.sort()
            lo = boot_scores[int(0.025 * n_boot)]
            hi = boot_scores[int(0.975 * n_boot) - 1]
            ci_text = f"Macro-F1 95% bootstrap CI (n_boot={n_boot}): [{lo:.4f}, {hi:.4f}]"

        # Append summary section to file
        f_out.write("\n\n" + "=" * 70 + "\n")
        f_out.write("=== FINAL METRICS SUMMARY ===\n")
        f_out.write(f"Accuracy: {acc:.4f}\n")
        f_out.write(f"Macro-F1: {maf:.4f}\n")
        if ci_text:
            f_out.write(ci_text + "\n")

        f_out.write("\n=== Prediction Distribution ===\n")
        for label in expected_labels + sorted([l for l in pred_counts.keys() if l not in expected_labels]):
            if label in pred_counts:
                count = pred_counts[label]
                f_out.write(f"  {label}: {count} ({count/len(preds)*100:.2f}%)\n")

        f_out.write("\n=== Ground Truth Distribution ===\n")
        for label in expected_labels + sorted([l for l in true_counts.keys() if l not in expected_labels]):
            if label in true_counts:
                count = true_counts[label]
                f_out.write(f"  {label}: {count} ({count/len(truth)*100:.2f}%)\n")

        f_out.write("\n=== Confusion Matrix (rows=true, cols=pred) ===\n")
        f_out.write("Labels: " + ", ".join(expected_labels) + "\n")
        f_out.write(str(cm) + "\n")

        f_out.write("\n=== Per-class Precision/Recall/F1/Support ===\n")
        for i, lbl in enumerate(expected_labels):
            f_out.write(
                f"  {lbl}: precision={prec[i]:.4f}, recall={rec[i]:.4f}, f1={f1s[i]:.4f}, support={support[i]}\n"
            )

        f_out.write("\n=== Classification Report ===\n")
        f_out.write(report + "\n")

        f_out.write("\n=== Error Rates per True Label ===\n")
        for idx, label in enumerate(expected_labels):
            total = cm[idx].sum()
            correct = cm[idx][idx]
            err = 1.0 - (correct / total) if total > 0 else 0.0
            f_out.write(f"  {label}: {err*100:.2f}%\n")

    # Also print concise console summary
    print("=== CaseHold Results ===")
    print(f"Saved detailed log to: {save_path.resolve()}")
    print(f"Accuracy = {acc:.4f}")
    print(f"Macro-F1 = {maf:.4f}")
    if ci_text:
        print(ci_text)

    return acc, maf, cm, predictions

def evaluate_billsum(
    model,
    tokenizer,
    device="cuda",
    n_eval=200,
    seed=1234,
    max_new_tokens=384,
    enable_thinking=False,
    batch_size=1,
    max_length=3000,  # IMPORTANT: define it
    save_output="assets/BillSum_generated_summaries.txt",
    think_token_id=None,               # auto-resolved from tokenizer; falls back to None
):
    rouge = load_metric("rouge")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    billsum_dataset = load_dataset("billsum", split="ca_test")

    texts = billsum_dataset["text"][:n_eval]
    titles = billsum_dataset["title"][:n_eval]
    reference_summaries = billsum_dataset["summary"][:n_eval]

    model.to(device)
    model.eval()

    # Ensure padding token exists for batched tokenization / generation
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has no pad_token_id and no eos_token_id; cannot pad safely.")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    tokenizer.padding_side = "left"  # for causal LMs, left padding is safer

    # Resolve think_token_id dynamically from tokenizer if not provided
    if enable_thinking:
        if think_token_id is None:
            _think_id = tokenizer.convert_tokens_to_ids("</think>")
            if _think_id is not None and _think_id != getattr(tokenizer, "unk_token_id", None):
                think_token_id = _think_id
            else:
                import warnings
                warnings.warn(
                    "enable_thinking=True but tokenizer has no '</think>' token. "
                    "Disabling thinking mode for this run."
                )
                enable_thinking = False

    generated_summaries = []
    start_time = time()

    save_path = Path(save_output)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f_out:
        f_out.write("=== BillSum Evaluation Log ===\n")
        f_out.write(f"seed={seed}, max_new_tokens={max_new_tokens}, n_eval={n_eval}\n")
        f_out.write(f"device={device}, batch_size={batch_size}, max_length={max_length}\n")
        f_out.write("=" * 70 + "\n\n")

        for start in tqdm(range(0, n_eval, batch_size), desc="Generating Summaries (batched)"):
            end = min(start + batch_size, n_eval)
            batch_texts = texts[start:end]
            batch_titles = titles[start:end]

            # Build prompts
            if enable_thinking:
                batch_prompts = []
                for t, x in zip(batch_titles, batch_texts):
                    p = billsum_input_template.format(title=t, input_text=x)
                    conv = [{"role": "user", "content": p}]
                    try:
                        batch_prompts.append(
                            tokenizer.apply_chat_template(
                                conv,
                                tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=True,
                            )
                        )
                    except TypeError:
                        # Tokenizer doesn't support enable_thinking (non-Qwen model)
                        batch_prompts.append(
                            tokenizer.apply_chat_template(
                                conv,
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                        )
                inputs = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(model.device)
            else:
                batch_prompts = [
                    billsum_input_template.format(title=t, input_text=x)
                    for t, x in zip(batch_titles, batch_texts)
                ]
                inputs = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(device)

            prompt_lens = inputs["attention_mask"].sum(dim=1)  # (B,)

            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample= True,
                    top_k=50,
                    top_p=0.9,
                    temperature=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            B, T = out.shape

            if not enable_thinking:
                # Slice off prompts (ragged); batch_decode handles decoding efficiently
                gen_seqs = [out[i, int(prompt_lens[i]):] for i in range(B)]
                batch_summaries = tokenizer.batch_decode(gen_seqs, skip_special_tokens=True)
                batch_summaries = [s.strip() for s in batch_summaries]
            else:
                # Vectorized find of last </think> token in generated continuation
                pos = torch.arange(T, device=out.device).unsqueeze(0).expand(B, T)
                gen_mask = pos >= prompt_lens.unsqueeze(1)
                is_think = (out == think_token_id) & gen_mask
                think_pos = torch.where(is_think, pos, torch.full_like(pos, -1))
                last_think_pos = think_pos.max(dim=1).values
                content_start = torch.where(last_think_pos >= 0, last_think_pos + 1, prompt_lens)

                gen_seqs = [out[i, int(content_start[i]):] for i in range(B)]
                batch_summaries = tokenizer.batch_decode(gen_seqs, skip_special_tokens=True)
                batch_summaries = [s.strip() for s in batch_summaries]

            generated_summaries.extend(batch_summaries)

            # Optional: per-example logging (correctly batched)
            for i, summary in enumerate(batch_summaries):
                global_idx = start + i
                f_out.write(f"\n=== Example #{global_idx} ===\n")
                f_out.write(f"TITLE:\n{titles[global_idx]}\n\n")
                # Uncomment if you want full text logging (files get huge)
                # f_out.write(f"TEXT:\n{texts[global_idx]}\n\n")
                f_out.write(f"REFERENCE SUMMARY:\n{reference_summaries[global_idx]}\n\n")
                f_out.write(f"GENERATED SUMMARY:\n{summary}\n")
                f_out.write("=" * 50 + "\n")

    # ROUGE compute
    rouge_scores = rouge.compute(predictions=generated_summaries, 
                                references=reference_summaries)

    # Robust printing across evaluate vs datasets.load_metric implementations
    print("\n=== Final ROUGE SCORES ===")
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")

    total_time = time() - start_time
    print(f"\nTotal generation and evaluation time: {total_time:.2f} seconds")

    return rouge_scores, generated_summaries


def evaluate_mednli(
    model,
    tokenizer,
    file_path="data/downloaded/physionet.org/files/mednli/1.0.0/mli_test_v1.jsonl",
    device="cuda",
    save_output="assets/mednli_predictions.txt",
    seed=1234,
    max_attempts=3,
    max_new_tokens=10,
    n_eval=None,              # optional: limit number of examples
    do_bootstrap_ci=True,
    n_boot=1000,
):
    """
    Evaluate MedNLI and save:
      - Per-example: id, sentence1, sentence2, gold, pred, raw generated continuation
      - End summary: accuracy, macro-F1, per-class metrics, confusion matrix, report, distributions, error rates,
                     optional bootstrap CI for macro-F1
    """

    random.seed(seed)
    torch.manual_seed(seed)

    save_path = Path(save_output)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Load JSONL
    with open(file_path, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]

    if n_eval is not None:
        data = data[: int(n_eval)]

    expected_labels = ["entailment", "contradiction", "neutral"]

    # Robust: first token id for each label string (for logits fallback)
    def first_token_id(text: str) -> int:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError(f"Tokenizer produced empty ids for label: {text}")
        return ids[0]

    label_token_ids = [first_token_id(lbl) for lbl in expected_labels]
    tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()

    ground_truth = []
    predictions = []

    with open(save_path, "w", encoding="utf-8") as f_out:
        f_out.write("=== MedNLI Evaluation Log ===\n")
        f_out.write(
            f"n_eval={len(data)}, seed={seed}, max_new_tokens={max_new_tokens}, max_attempts={max_attempts}\n"
        )
        f_out.write(f"device={device}\n")
        f_out.write("=" * 70 + "\n\n")

        for idx, entry in enumerate(tqdm(data, desc="Evaluating MedNLI")):
            sentence1 = entry["sentence1"]
            sentence2 = entry["sentence2"]
            gold_label = str(entry["gold_label"]).strip().lower()

            prompt = mednli_input_template.format(sentence1=sentence1, sentence2=sentence2)

            prediction = "unknown"
            decoded_full = ""
            raw_generation = ""

            for attempt in range(max_attempts):
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=2048,
                ).to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        top_k=50,
                        top_p=0.9,
                        temperature=0.9,
                        output_scores=True,
                        return_dict_in_generate=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                outputs = outputs.sequences[0][len(inputs.input_ids[0])-10:].tolist()
                decoded_output = tokenizer.decode(outputs, skip_special_tokens=True).lower()
                match = re.search(
                            r"\bTheir relationship is\b\s*[:\-]?\s*"
                            r"(?:\*\*|\*)?\s*"          # optional markdown emphasis opening
                            r"(?:['\"]{1,2})?\s*"       # optional quote wrapper: ' or " or '' or ""
                            r"(entailment|contradiction|neutral)"
                            r"\s*(?:['\"]{1,2})?\s*"    # optional closing quotes
                            r"(?:\*\*|\*)?\b",          # optional markdown emphasis closing
                            decoded_output,
                            flags=re.IGNORECASE
                        )
                if match:
                    prediction = match.group(1).strip()
                    break


            predictions.append(prediction)
            ground_truth.append(gold_label)

            # Per-example logging
            example_id = entry.get("pairID", entry.get("id", idx))
            f_out.write(f"example_id: {example_id}\n")
            f_out.write(f"GOLD: {gold_label}\n")
            f_out.write(f"PRED: {prediction}\n")
            f_out.write("SENTENCE1:\n")
            f_out.write(sentence1.strip() + "\n")
            f_out.write("SENTENCE2:\n")
            f_out.write(sentence2.strip() + "\n")
            f_out.write("RAW_GENERATION:\n")
            matches = list(re.finditer(r"\bFinal Answer\s*:\s*", decoded_output, flags=re.IGNORECASE))
            if matches:
                last_match = matches[-1]
                raw_generation = decoded_output[last_match.end():].strip()
            f_out.write(raw_generation.strip() + "\n")
            f_out.write("-" * 70 + "\n\n")

        # Metrics
        accuracy = accuracy_score(ground_truth, predictions)
        macro_f1 = f1_score(ground_truth, predictions, average="macro", labels=expected_labels)

        cm = confusion_matrix(ground_truth, predictions, labels=expected_labels)

        prec, rec, f1s, support = precision_recall_fscore_support(
            ground_truth, predictions, labels=expected_labels, zero_division=0
        )

        pred_counts = Counter(predictions)
        true_counts = Counter(ground_truth)

        report = classification_report(
            ground_truth, predictions, labels=expected_labels, zero_division=0
        )

        # Optional bootstrap CI for macro-F1
        ci_text = ""
        if do_bootstrap_ci:
            rng = random.Random(seed)
            n = len(ground_truth)
            boot_scores = []
            for _ in range(n_boot):
                idxs = [rng.randrange(n) for _ in range(n)]
                t_b = [ground_truth[i] for i in idxs]
                p_b = [predictions[i] for i in idxs]
                boot_scores.append(f1_score(t_b, p_b, average="macro", labels=expected_labels))
            boot_scores.sort()
            lo = boot_scores[int(0.025 * n_boot)]
            hi = boot_scores[int(0.975 * n_boot) - 1]
            ci_text = f"Macro-F1 95% bootstrap CI (n_boot={n_boot}): [{lo:.4f}, {hi:.4f}]"

        # Append final metrics to the same file
        f_out.write("\n\n" + "=" * 70 + "\n")
        f_out.write("=== FINAL METRICS SUMMARY ===\n")
        f_out.write(f"Accuracy: {accuracy:.4f}\n")
        f_out.write(f"Macro-F1: {macro_f1:.4f}\n")
        if ci_text:
            f_out.write(ci_text + "\n")

        f_out.write("\n=== Prediction Distribution ===\n")
        for label in expected_labels + sorted([l for l in pred_counts.keys() if l not in expected_labels]):
            if label in pred_counts:
                count = pred_counts[label]
                f_out.write(f"  {label}: {count} ({count/len(predictions)*100:.2f}%)\n")

        f_out.write("\n=== Ground Truth Distribution ===\n")
        for label in expected_labels + sorted([l for l in true_counts.keys() if l not in expected_labels]):
            if label in true_counts:
                count = true_counts[label]
                f_out.write(f"  {label}: {count} ({count/len(ground_truth)*100:.2f}%)\n")

        f_out.write("\n=== Confusion Matrix (rows=true, cols=pred) ===\n")
        f_out.write("Labels: " + ", ".join(expected_labels) + "\n")
        f_out.write(str(cm) + "\n")

        f_out.write("\n=== Per-class Precision/Recall/F1/Support ===\n")
        for i, lbl in enumerate(expected_labels):
            f_out.write(
                f"  {lbl}: precision={prec[i]:.4f}, recall={rec[i]:.4f}, f1={f1s[i]:.4f}, support={support[i]}\n"
            )

        f_out.write("\n=== Classification Report ===\n")
        f_out.write(report + "\n")

        f_out.write("\n=== Error Rates per True Label ===\n")
        for i, lbl in enumerate(expected_labels):
            total = cm[i].sum()
            correct = cm[i][i]
            err = 1.0 - (correct / total) if total > 0 else 0.0
            f_out.write(f"  {lbl}: {err*100:.2f}%\n")

    print("=== MedNLI Results ===")
    print(f"Saved detailed log to: {save_path.resolve()}")
    print(f"Accuracy = {accuracy:.4f}")
    print(f"Macro-F1 = {macro_f1:.4f}")
    if ci_text:
        print(ci_text)

    return accuracy, macro_f1, cm, predictions


def evaluate_pubmedqa(
    model,
    tokenizer,
    device="cuda",
    save_output="assets/pubmedqa_predictions.txt",
    seed=1234,
    n_eval=500,
    max_new_tokens=10,
    max_attempts=3,
    do_bootstrap_ci=True,
    n_boot=1000,
):
    """
    Evaluate a model on the first `n_eval` instances of PubMedQA (pqa_labeled/train).

    Saves to `save_output`:
      - For each example: pubid, question, gold, pred, raw decoded output
      - At end: accuracy, macro-F1, per-class metrics, confusion matrix, report, distributions,
                and optional bootstrap CI for macro-F1.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    # Ensure output dir exists
    save_path = Path(save_output)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    tokenizer.pad_token = tokenizer.eos_token

    ground_truth = {row["pubid"]: str(row["final_decision"]).strip().lower() for row in data}
    questions = {row["pubid"]: row["question"] for row in data}
    contexts_data = {row["pubid"]: row["context"] for row in data}

    expected_labels = ["yes", "no", "maybe"]

    # Robust: get first token id of each label string
    def first_token_id(text: str) -> int:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError(f"Tokenizer produced empty ids for label: {text}")
        return ids[0]

    label_token_ids = [first_token_id(lbl) for lbl in expected_labels]

    pubids = list(questions.keys())[:n_eval]

    predictions = {}
    raw_outputs = {}

    model.to(device)
    model.eval()

    # Open output file once; write header + per-sample logs as we go
    with open(save_path, "w", encoding="utf-8") as f_out:
        f_out.write("=== PubMedQA Evaluation Log ===\n")
        f_out.write(f"n_eval={n_eval}, seed={seed}, max_new_tokens={max_new_tokens}, max_attempts={max_attempts}\n")
        f_out.write(f"device={device}\n")
        f_out.write("=" * 70 + "\n\n")

        for pubid in tqdm(pubids, desc="Predicting final decisions"):
            question = questions[pubid]
            context_info = contexts_data[pubid] or {}

            contexts_text = "\n\n".join(context_info.get("contexts", []) or [])
            labels_text = ", ".join(context_info.get("labels", []) or [])
            meshes_text = ", ".join(context_info.get("meshes", []) or [])

            prompt = pubmed_input_template.format(
                contexts=contexts_text,
                labels=labels_text,
                meshes=meshes_text,
                question=question,
            )

            prediction = "unknown"
            decoded_output = ""

            for attempt in range(max_attempts):
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=2048,
                ).to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        top_k=50,
                        top_p=0.9,
                        temperature=0.9,
                        output_scores=True,
                        return_dict_in_generate=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                outputs = outputs.sequences[0][len(inputs.input_ids[0])-10:].tolist()
                decoded_output = tokenizer.decode(outputs, skip_special_tokens=True).lower()
                match = re.search(
                            r"\bthe answer is\b\s*[:\-]?\s*"
                            r"(?:\*\*|\*)?\s*"          # optional markdown emphasis opening
                            r"(?:['\"]{1,2})?\s*"       # optional quote wrapper: ' or " or '' or ""
                            r"(yes|no|maybe)"
                            r"\s*(?:['\"]{1,2})?\s*"    # optional closing quotes
                            r"(?:\*\*|\*)?\b",          # optional markdown emphasis closing
                            decoded_output,
                            flags=re.IGNORECASE
                        )
                if match:
                    prediction = match.group(1).strip()
                    break

            gold = ground_truth.get(pubid, "unknown")

            predictions[pubid] = prediction
            raw_outputs[pubid] = decoded_output

            # Write per-example block
            f_out.write(f"pubid: {pubid}\n")
            f_out.write(f"GOLD: {gold}\n")
            f_out.write(f"PRED: {prediction}\n")
            f_out.write("QUESTION:\n")
            f_out.write(question.strip() + "\n")
            f_out.write("RAW_DECODED:\n")
            # matches = list(re.finditer(r"\bresponse\s*:\s*", decoded_output, flags=re.IGNORECASE))
            # if matches:
            #     last_match = matches[-1]
            #     decoded_output = decoded_output[last_match.end():].strip()
            f_out.write(decoded_output.strip() + "\n")
            f_out.write("-" * 70 + "\n\n")

        # Compute metrics
        truth = [ground_truth[pmid] for pmid in pubids]
        preds = [predictions[pmid] for pmid in pubids]

        acc = accuracy_score(truth, preds)
        maf = f1_score(truth, preds, average="macro", labels=expected_labels)

        cm = confusion_matrix(truth, preds, labels=expected_labels)

        # Per-class metrics
        prec, rec, f1s, support = precision_recall_fscore_support(
            truth, preds, labels=expected_labels, zero_division=0
        )

        pred_counts = Counter(preds)
        true_counts = Counter(truth)

        report = classification_report(
            truth, preds, labels=expected_labels, zero_division=0
        )

        # Optional bootstrap CI for macro-F1
        ci_text = ""
        if do_bootstrap_ci:
            rng = random.Random(seed)
            n = len(truth)
            boot_scores = []
            for _ in range(n_boot):
                idxs = [rng.randrange(n) for _ in range(n)]
                t_b = [truth[i] for i in idxs]
                p_b = [preds[i] for i in idxs]
                boot_scores.append(f1_score(t_b, p_b, average="macro", labels=expected_labels))
            boot_scores.sort()
            lo = boot_scores[int(0.025 * n_boot)]
            hi = boot_scores[int(0.975 * n_boot) - 1]
            ci_text = f"Macro-F1 95% bootstrap CI (n_boot={n_boot}): [{lo:.4f}, {hi:.4f}]"

        # Append summary section to file
        f_out.write("\n\n" + "=" * 70 + "\n")
        f_out.write("=== FINAL METRICS SUMMARY ===\n")
        f_out.write(f"Accuracy: {acc:.4f}\n")
        f_out.write(f"Macro-F1: {maf:.4f}\n")
        if ci_text:
            f_out.write(ci_text + "\n")

        f_out.write("\n=== Prediction Distribution ===\n")
        for label in expected_labels + sorted([l for l in pred_counts.keys() if l not in expected_labels]):
            if label in pred_counts:
                count = pred_counts[label]
                f_out.write(f"  {label}: {count} ({count/len(preds)*100:.2f}%)\n")

        f_out.write("\n=== Ground Truth Distribution ===\n")
        for label in expected_labels + sorted([l for l in true_counts.keys() if l not in expected_labels]):
            if label in true_counts:
                count = true_counts[label]
                f_out.write(f"  {label}: {count} ({count/len(truth)*100:.2f}%)\n")

        f_out.write("\n=== Confusion Matrix (rows=true, cols=pred) ===\n")
        f_out.write("Labels: " + ", ".join(expected_labels) + "\n")
        f_out.write(str(cm) + "\n")

        f_out.write("\n=== Per-class Precision/Recall/F1/Support ===\n")
        for i, lbl in enumerate(expected_labels):
            f_out.write(
                f"  {lbl}: precision={prec[i]:.4f}, recall={rec[i]:.4f}, f1={f1s[i]:.4f}, support={support[i]}\n"
            )

        f_out.write("\n=== Classification Report ===\n")
        f_out.write(report + "\n")

        f_out.write("\n=== Error Rates per True Label ===\n")
        for idx, label in enumerate(expected_labels):
            total = cm[idx].sum()
            correct = cm[idx][idx]
            err = 1.0 - (correct / total) if total > 0 else 0.0
            f_out.write(f"  {label}: {err*100:.2f}%\n")

    # Also print concise console summary
    print("=== PubMedQA Results ===")
    print(f"Saved detailed log to: {save_path.resolve()}")
    print(f"Accuracy = {acc:.4f}")
    print(f"Macro-F1 = {maf:.4f}")
    if ci_text:
        print(ci_text)

    return acc, maf, cm, predictions

def evaluate_hqs(
    model,
    tokenizer,
    file_path="data/downloaded/MEDIQA2021-Task1-TestSet-ReferenceSummaries.xlsx",
    save_output="assets/HQS_generated_summaries.txt",
    max_new_tokens=50,
    enable_thinking=False,
    think_token_id=None,
    device="cuda",
    n_eval=100,
    seed=1234,
):
    """
    Summarize each question in an Excel file using a simple prompt and compute ROUGE.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    # Resolve think_token_id dynamically from tokenizer if not provided
    if enable_thinking:
        if think_token_id is None:
            _think_id = tokenizer.convert_tokens_to_ids("</think>")
            if _think_id is not None and _think_id != getattr(tokenizer, "unk_token_id", None):
                think_token_id = _think_id
            else:
                import warnings
                warnings.warn(
                    "enable_thinking=True but tokenizer has no '</think>' token. "
                    "Disabling thinking mode for this run."
                )
                enable_thinking = False
    rouge = load_metric("rouge")
    df = pd.read_excel(file_path)

    questions = df["NLM Question"].tolist()
    reference_summaries = df["Summary"].tolist()

    # ---- NEW: cap eval set size via n_eval ----
    total_available = len(questions)
    if n_eval is None or (isinstance(n_eval, int) and n_eval <= 0):
        eval_n = total_available
    else:
        eval_n = min(int(n_eval), total_available)

    questions = questions[:eval_n]
    reference_summaries = reference_summaries[:eval_n]
    # ------------------------------------------

    model.to(device)
    model.eval()

    generated_summaries = []
    start_time = time()

    save_path = Path(save_output)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Open a file to log question/summary pairs
    with open(save_path, "w", encoding="utf-8") as f_out:
        f_out.write("=== HQS Evaluation Log ===\n")
        f_out.write(
            f"seed={seed}, max_new_tokens={max_new_tokens}, n_eval={eval_n}/{total_available}\n"
        )
        f_out.write(f"device={device}\n")
        f_out.write("=" * 70 + "\n\n")

        for idx, question in enumerate(tqdm(questions, desc="Generating Summaries")):
            if enable_thinking:
                prompt = hqs_input_template.format(input_question=question)
                prompt = [{"role": "user", "content": prompt}]
                try:
                    prompt = tokenizer.apply_chat_template(
                        prompt,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True,
                    )
                except TypeError:
                    # Tokenizer doesn't support enable_thinking (non-Qwen model)
                    prompt = tokenizer.apply_chat_template(
                        prompt,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
            else:
                prompt = hqs_input_template.format(input_question=question)
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=1024,
                    truncation=True,
                ).to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    temperature=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            if enable_thinking:
                output_ids = output_ids[0][len(inputs.input_ids[0]) :].tolist()
                try:
                    # rindex finding </think> token
                    index = len(output_ids) - output_ids[::-1].index(think_token_id)
                except ValueError:
                    index = 0

                thinking_content = tokenizer.decode(
                    output_ids[:index], skip_special_tokens=True
                ).strip("\n")
                content = tokenizer.decode(
                    output_ids[index:], skip_special_tokens=True
                ).strip("\n")
                summary = clean_first_sentence(content)
            else:
                decoded_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
                # Remove the prompt from the output if it is echoed back
                summary = decoded_text[len(prompt) :].strip()
                summary = clean_first_sentence(summary)

            generated_summaries.append(summary)

            log_entry = (
                f"\n=== Generated Summary for Question #{idx+1} ===\n"
                f"QUESTION:\n{question}\n"
                f"GENERATED SUMMARY:\n{summary}\n"
                + "=" * 50
                + "\n"
            )
            f_out.write(log_entry)

    rouge_scores = rouge.compute(
        predictions=generated_summaries,
        references=reference_summaries,
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
        if idx == -1:
            continue

        if idx == 0:
            # Marker is at the very start: remove the marker itself, keep the rest
            t = t[len(m):].lstrip()
            # keep scanning in case there are multiple leading markers
            continue

        # Marker appears later: drop everything starting from the marker
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