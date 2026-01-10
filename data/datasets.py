import os
import json
import random
from datasets import load_dataset
import pandas as pd
from data.templates import *

local_data_directory = "data/downloaded"

def get_loaders(
    name,
    size = 1,
    nsamples=128,
    seed=0,
    seqlen=2048,
    tokenizer=None,
    max_attempts=10000,
    padding=True
):
    """
    Dynamically choose which dataset to load. 
    'padding' argument is passed to those loaders that support it.
    """
    tokenizer.padding_side = "left"
    if 'wikitext2' in name:
        # wikitext2 does not have a padding argument
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if 'harrison' in name:
        # Harrison does not have a padding argument
        return get_Harrison(nsamples, seed, seqlen, tokenizer)
    if 'pplmultilegal' in name:
        return get_multilegalpile_ppl(nsamples, seed, seqlen, tokenizer)
    if 'c4' in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    raise ValueError(f"Unknown dataset name '{name}'")

def construct_med_data(pubmed_count=7000,
                       pubmed_yes=0.5,
                       pubmed_no=0.5,
                       mednli_count=7000,
                       mednli_entailment=0.33,
                       mednli_contradiction=0.33,
                       hqs_count=1000,
                       c4_count=0,
                       seed=1234,
                       split='train'):
    """
    Constructs a list of fine-tuning examples from four sources:
      - PubMedQA (formatted in Alcapa style)
      - MedNLI (formatted in Alcapa style)
      - HQS/MeQSum (formatted in Alcapa style)
      - C4 (kept in standard format)
    
    Each example is a dict with keys: 'source', 'input_text', and 'target_text'.
    
    Parameters:
      - pubmed_count (int): number of PubMedQA examples to retrieve.
      - pubmed_yes (float): proportion of positive examples for PubMedQA.
      - pubmed_no (float): proportion of negative examples for PubMedQA.
      - mednli_count (int): number of MedNLI examples to retrieve.
      - mednli_entailment (float): proportion of entailment examples.
      - mednli_contradiction (float): proportion of contradiction examples.
      - hqs_count (int): number of HQS/MeQSum examples to use.
      - c4_count (int): number of C4 examples to use.
      - seed (int): random seed for reproducibility.
    
    Returns:
      - A list of dictionaries, each with keys "source", "input_text", and "target_text".
    """
    random.seed(seed)
    finetune_data = []

    # Retrieve PubMedQA data. (Assumes get_pubmedqa is defined elsewhere)
    pubmedqa_data = get_pubmedqa(pubmed_count=pubmed_count, 
                                 positive=pubmed_yes, 
                                 negative=pubmed_no, 
                                 seed=seed,
                                 split=split)

    # Retrieve MedNLI data. (Assumes get_mednli is defined elsewhere)
    mednli_data = get_mednli(mednli_count=mednli_count, 
                             entailment_prop=mednli_entailment, 
                             contradiction_prop=mednli_contradiction, 
                             seed=seed,
                             split=split)

    # --- HQS / MeQSum (Alcapa format) ---
    hqs_data = get_hqs(hqs_count=hqs_count,
                       seed=seed,
                       split=split)
    
    # --- C4 Dataset (Standard format) ---
    # For C4, we simply include the raw text as is.
    if c4_count > 0:
        c4_data = load_dataset('allenai/c4', 'en', split='train', streaming=True)
        c4_examples = []
        for example in c4_data:
            text = example['text']
            # Optionally skip very short texts.
            if len(text) < 100:
                continue
            c4_examples.append({
                "source": "C4",
                "input_text": text,
                "target_text": ""
            })
            if len(c4_examples) >= c4_count:
                break
        finetune_data.extend(c4_examples)
    
    # Extend with MedNLI and PubMedQA data.
    finetune_data.extend(mednli_data)
    finetune_data.extend(pubmedqa_data)
    finetune_data.extend(hqs_data)
    
    # Shuffle the examples.
    random.shuffle(finetune_data)
    
    return finetune_data

def construct_legal_data(casehold_count=7000, 
                         casehold_prop = [1/5, 1/5, 1/5, 1/5, 1/5], 
                         billsum_count=2000, 
                         contractnli_count=6000,
                         entailment_prop=0.35, 
                         contradiction_prop=0.2,  
                         c4_count=0, 
                         seed=1234,
                         split='train'):
    """
    Constructs a list of fine-tuning examples from three sources for the legal domain.
    
    The data instances follow the Alpaca (Taori et al., 2023) template so that models are 
    trained to predict the responses.
    
    Sources:
      - CaseHOLD: 13000 training instances.
      - BillSum: 2000 training instances.
      - C4: a configurable number of examples from the general domain.
    
    Each example is a dict with keys: 'source', 'input_text', and 'target_text'.
    """
    finetune_data = []
    random.seed(seed)
    
    # --- CaseHOLD (Alpaca-style) ---
    # Load streaming dataset; iterate and collect 13000 examples.
    casehold_dataset = get_casehold(casehold_count=casehold_count, 
                                    choice_prop = casehold_prop, 
                                    seed=seed,
                                    split=split)

    # --- BillSum (Alpaca-style) ---
    billsum_dataset = get_billsum(billsum_count=billsum_count, 
                                  seed=seed,
                                  split=split)
    # --- ContractNLI (Alpaca-style) ---
    contractnli = get_contractnli(contractnli_count=contractnli_count,
                                    entailment_prop=entailment_prop, 
                                    contradiction_prop=contradiction_prop,  
                                    seed=seed,
                                    split=split)

    # --- C4 Dataset (Standard format) ---
    if split == 'train':
        c4_dataset = load_dataset('allenai/c4', 'en', split='train', streaming=True)
        c4_examples = 0
        for example in c4_dataset:
            text = example.get('text', "")
            if len(text) < 100:
                continue
            finetune_data.append({
                "source": "C4",
                "input_text": text,  # Raw text.
                "target_text": ""    # No additional formatting.
            })
            c4_examples += 1
            if c4_examples >= c4_count:
                break

    finetune_data.extend(casehold_dataset)
    finetune_data.extend(billsum_dataset)
    finetune_data.extend(contractnli)
    random.shuffle(finetune_data)

    return finetune_data

def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_Harrison(nsamples, seed, seqlen, tokenizer):
    """
    Prepare the Harrison dataset for perplexity evaluation.

    Parameters:
        nsamples: int
            Number of samples to generate for the training set.
        seed: int
            Seed for reproducibility.
        seqlen: int
            Sequence length for evaluation.
        tokenizer: Hugging Face tokenizer
            Tokenizer to process text data.

    Returns:
        Tuple[List[Tuple[Tensor, Tensor]], Tensor]:
            trainloader: List of input-target pairs for training.
            testenc: Encoded dataset used for full evaluation.
    """
    # Load the dataset
    traindata = load_dataset(
        'cogbuji/medqa_corpus_en', 
        'core_clinical', 
        'InternalMed_Harrison.txt', 
        split='train',
        trust_remote_code=True
    )
    # Filter for Harrison textbook
    traindata = traindata.filter(
        lambda example: example["source"] == "textbooks/en/InternalMed_Harrison.txt"
    )

    encoded_text = tokenizer(" ".join(traindata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, encoded_text.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = encoded_text.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100  # Mask tokens except the last one
        trainloader.append((inp, tar))

    return trainloader, encoded_text


def get_multilegalpile_ppl(nsamples, seed, seqlen, tokenizer):
    """
    Prepare a subset of the Multi-Legal-Pile dataset (first 300 items) for perplexity evaluation.

    Args:
        nsamples (int): Number of samples to generate for the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for evaluation.
        tokenizer: HuggingFace tokenizer to process text data.

    Returns:
        (trainloader, encoded_text):
            trainloader (List[Tuple[Tensor, Tensor]]): A list of (inp, tar) pairs 
                for training, each of shape [1, seqlen].
            encoded_text (BatchEncoding): Encoded dataset (only the first 300 items) 
                used for full evaluation.
    """
    # 1. Load the dataset in streaming mode, filtered by jurisdiction='US'
    config = 'en_legislation'
    traindata = load_dataset(
        'joelniklaus/Multi_Legal_Pile',
        config,
        split='train',
        streaming=True,
        trust_remote_code=True
    )
    # traindata = traindata.filter(lambda example: example["jurisdiction"] == "US")

    # 2. Collect the FIRST 300 samples from the streaming dataset
    texts = []
    for i, example in enumerate(traindata):
        if i >= 300:
            break
        texts.append(example["text"])

    # 3. Tokenize all 300 samples as a single joined string
    encoded_text = tokenizer(" ".join(texts), return_tensors='pt')

    # 4. Generate nsamples from this encoded text
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        # Randomly choose a starting index
        i = random.randint(0, encoded_text.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = encoded_text.input_ids[:, i:j]
        tar = inp.clone()
        # Mask out everything except the last token in each sequence
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, encoded_text

class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

def get_c4(nsamples, seed, seqlen, tokenizer):
    """
    Returns:
      trainloader: a list of (input_ids, target_ids) pairs. 
                   Each is shape [seq_len], suitable for a DataLoader.
      valdata:     a TokenizerWrapper with shape [val_seq_len],
                   or however you wish to use the validation text.
    """
    # Load train and validation datasets
    traindata = load_dataset(
        'allenai/c4',  
        data_files='en/c4-train.00000-of-01024.json.gz',
        split='train'
    )
    valdata = load_dataset(
        'allenai/c4', 
        data_files='en/c4-validation.00000-of-00008.json.gz', 
        split='train'
    )

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            # Pick a random example from the dataset
            idx = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[idx]['text'], return_tensors='pt')
            # Check if there's enough length for seqlen
            if trainenc.input_ids.shape[1] > seqlen:
                break
        
        # Now randomly pick a slice of length seqlen
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        
        # Here we squeeze out the batch dimension => shape [seqlen]
        inp = trainenc.input_ids[0, i:j]  # shape [seqlen]
        
        # Create targets
        tar = inp.clone()  # shape [seqlen]
        tar[:-1] = -100    # only the last token is predicted, etc.
        
        trainloader.append((inp, tar))

    # Prepare validation dataset
    # We join some subset of valdata text into one big string
    val_text = ' '.join(valdata[:1100]['text'])
    valenc = tokenizer(val_text, return_tensors='pt')
    
    # Slice to a max of 256 * seqlen
    valenc = valenc.input_ids[0, : (256 * seqlen)]  # shape [val_seq_len]

    # Wrap in a helper class or just return the tensor
    valdata = TokenizerWrapper(valenc)

    return trainloader, valdata

def get_pubmedqa(
    pubmed_count=7000,
    positive=0.5,
    negative=0.3,
    seed=1234,
    file_path="data/downloaded/pubmedqa_maybe_output.json",
    split='train'
):
    """
    Prepare fine-tuning data from PubMedQA dataset with properly formatted prompts.
    Sampling positive (yes), negative (no), and maybe examples according to specified proportions.
    """
    finetune_data = []
    random.seed(seed)

    # Load the PubMedQA dataset
    if split == 'train':
        pubmed_data = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
    elif split == 'validation':
        pubmed_data = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
        pubmed_data = pubmed_data.shuffle(seed=seed).select(range(pubmed_count))
    else:
        raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'validation'.")
    
    if len(pubmed_data) > pubmed_count:
        pubmed_data = pubmed_data.select(range(pubmed_count))

    # Load optional maybe data from file (robust to missing/invalid files)
    maybe_data = []
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                maybe_data = loaded
            else:
                print(f"[get_pubmedqa] maybe file exists but is not a list; ignoring: {file_path}")
                maybe_data = []
        except Exception as e:
            print(f"[get_pubmedqa] failed to load maybe file; ignoring: {file_path}. Error: {e}")
            maybe_data = []
    else:
        print(f"[get_pubmedqa] no maybe data at {file_path}")

    # Group the data by final decision (yes, no, maybe)
    yes_data = []
    no_data = []

    for row in pubmed_data:
        label = str(row.get("final_decision", "")).strip().lower()
        if label == "yes":
            yes_data.append(row)
        elif label == "no":
            no_data.append(row)
        elif label == "maybe":
            maybe_data.append(row)


    if positive + negative > 1.0:
        raise ValueError("The sum of positive and negative proportions must be <= 1.0.")

    desired_yes = int(pubmed_count * positive)
    desired_no = int(pubmed_count * negative)
    desired_maybe = pubmed_count - desired_yes - desired_no

    def sample_data(data_list, desired_count):
        """Safe sampler: returns [] if desired_count==0 or data_list is empty."""
        if desired_count <= 0:
            return []
        if not data_list:
            return []
        if len(data_list) >= desired_count:
            return random.sample(data_list, desired_count)
        return random.choices(data_list, k=desired_count)

    sampled_yes = sample_data(yes_data, desired_yes)
    sampled_no = sample_data(no_data, desired_no)
    sampled_maybe = sample_data(maybe_data, desired_maybe)

    # If maybe_data is unavailable, fill the remainder from yes/no so we still hit pubmed_count
    final_samples = sampled_yes + sampled_no + sampled_maybe
    remaining = pubmed_count - len(final_samples)
    if remaining > 0:
        fallback_pool = yes_data + no_data
        if fallback_pool:
            print(f"[get_pubmedqa] filling {remaining} missing samples from yes/no (maybe unavailable/insufficient).")
            final_samples += sample_data(fallback_pool, remaining)
        else:
            print("[get_pubmedqa] warning: no data available to fill remaining samples.")
    else:
        # In case of any overfill (shouldn't happen with the logic above, but keep it safe)
        final_samples = final_samples[:pubmed_count]

    random.shuffle(final_samples)

    for row in final_samples:
        # Handle both HF dataset rows and plain dicts from JSON
        context_obj = row.get("context", {})
        if isinstance(context_obj, dict):
            contexts_list = context_obj.get("contexts", [])
            contexts = "\n\n".join(contexts_list) if isinstance(contexts_list, list) else str(contexts_list)
            labels = ", ".join(context_obj.get("labels", []) or [])
            meshes = ", ".join(context_obj.get("meshes", []) or [])
        else:
            contexts = str(context_obj) if context_obj is not None else ""
            labels = ""
            meshes = ""

        input_text = pubmed_input_template.format(
            contexts=contexts,
            labels=labels,
            meshes=meshes,
            question=row.get("question", "")
        )

        target_text = pubmed_target_template.format(
            ground_truth=str(row.get("final_decision", "")).strip().lower()
        )

        finetune_data.append({
            "source": "PubMedQA",
            "input_text": input_text,
            "target_text": target_text
        })

    return finetune_data

def get_mednli(mednli_count=7000, 
               entailment_prop=0.33, 
               contradiction_prop=0.33, 
               seed = 1234,
               split = 'train',
               mednli_file_path = 'data/downloaded/'):
    random.seed(seed)
    finetune_data = []

    if split == 'train':
        mednli_file_path = os.path.join(mednli_file_path, 'physionet.org/files/mednli/1.0.0/mli_train_v1.jsonl')
    elif split == 'validation':
        mednli_file_path = os.path.join(mednli_file_path, 'physionet.org/files/mednli/1.0.0/mli_dev_v1.jsonl')
    else:
        raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'validation'.")
    
    with open(mednli_file_path, "r", encoding="utf-8") as file:
        mednli_data = [json.loads(line) for line in file]
    
    # Calculate the neutral proportion.
    neutral_prop = 1 - entailment_prop - contradiction_prop
    if neutral_prop < 0:
        raise ValueError("The sum of entailment_prop and contradiction_prop cannot exceed 1.")
    
    # Filter the data by each category.
    data_by_category = {
        "entailment": [entry for entry in mednli_data if entry["gold_label"] == "entailment"],
        "contradiction": [entry for entry in mednli_data if entry["gold_label"] == "contradiction"],
        "neutral": [entry for entry in mednli_data if entry["gold_label"] == "neutral"]
    }
    
    # Determine the number of instances for each category.
    required_counts = {
        "entailment": int(round(mednli_count * entailment_prop)),
        "contradiction": int(round(mednli_count * contradiction_prop)),
        "neutral": int(round(mednli_count * neutral_prop))
    }
    
    # Adjust counts in case of rounding issues.
    total_required = sum(required_counts.values())
    if total_required != mednli_count:
        diff = mednli_count - total_required
        # Adjust the category with the highest proportion.
        max_label = max(
            ("entailment", entailment_prop),
            ("contradiction", contradiction_prop),
            ("neutral", neutral_prop),
            key=lambda x: x[1]
        )[0]
        required_counts[max_label] += diff
    
    # Sample the specified number of entries from each category.
    sampled_entries = []
    for label, count in required_counts.items():
        available = data_by_category[label]
        if len(available) < count:
            print(f"Warning: Only {len(available)} available for label '{label}', needed {count}. Using all available entries.")
            count = len(available)
        sampled_entries.extend(random.sample(available, count))
    
    # Shuffle the combined entries.
    random.shuffle(sampled_entries)
    
    # Build the finetune data using the templates.
    for entry in sampled_entries:
        input_text = mednli_input_template.format(
            sentence1=entry["sentence1"],
            sentence2=entry["sentence2"]
        )
        target_text = mednli_target_template.format(
            gold_label=entry["gold_label"]
        )
        finetune_data.append({
            "source": "MedNLI",
            "input_text": input_text,
            "target_text": target_text
        })
        
    return finetune_data

def get_hqs(hqs_count = 1000, 
            seed = 1234,
            split = 'train',
            data_dir = 'data/downloaded/MeQSum_ACL2019_BenAbacha_Demner-Fushman.xlsx'):
    
    random.seed(seed)
    
    finetune_data = []
    # Load the MeQSum dataset from Excel.
    hqs_df = pd.read_excel(data_dir)
    if split == 'train':
        hqs_df = hqs_df.head(hqs_count)
    elif split == 'validation':
        hqs_df = hqs_df.sample(frac=1, random_state=seed).head(hqs_count)
    
    for _, row in hqs_df.iterrows():
        input_text = hqs_input_template.format(input_question=row['CHQ'])
        target_text = hqs_target_template.format(sum=row['Summary'])  # Adjust column name as needed.
        finetune_data.append({
            "source": "MeQSum",
            "input_text": input_text,
            "target_text": target_text
        })

    return finetune_data

def get_casehold(casehold_count=7000, 
                 choice_prop=[1/5, 1/5, 1/5, 1/5, 1/5],
                 seed=1234,
                 split='train'):
    random.seed(seed)
    
    finetune_data = []
    casehold_dataset = load_dataset('casehold/casehold', split=split, trust_remote_code=True)
    
    # Assume the labels are strings "0", "1", "2", "3", "4".
    labels = ["0", "1", "2", "3", "4"]
    
    # Normalize the proportions (in case they do not sum to 1) and determine quotas per label.
    total_prop = sum(choice_prop)
    normalized_props = [p/total_prop for p in choice_prop]
    designated_counts = {label: int(round(casehold_count * normalized_props[i])) for i, label in enumerate(labels)}
    
    # Adjust if rounding caused a mismatch in total count.
    current_sum = sum(designated_counts.values())
    diff = casehold_count - current_sum
    for i in range(diff):
        designated_counts[labels[i % len(labels)]] += 1

    # Initialize counters for each label.
    counts = {label: 0 for label in labels}
    total_added = 0
    
    for example in casehold_dataset:
        try:
            citing_prompt = example['citing_prompt']
            holding_0 = example.get('holding_0', "")
            holding_1 = example.get('holding_1', "")
            holding_2 = example.get('holding_2', "")
            holding_3 = example.get('holding_3', "")
            holding_4 = example.get('holding_4', "")
            label = example['label']  # label is expected to be a string like "0", "1", etc.
        except KeyError:
            continue
        
        # Only add if the label is one of our designated labels and its quota is not met.
        if label not in counts:
            continue
        
        if counts[label] < designated_counts[label]:
            formatted_text = casehold_input_template.format(
                citing_prompt=citing_prompt,
                holding_0=holding_0,
                holding_1=holding_1,
                holding_2=holding_2,
                holding_3=holding_3,
                holding_4=holding_4,
            )
            target_text = casehold_target_template.format(label=label)
            finetune_data.append({
                "source": "CaseHOLD",
                "input_text": formatted_text,
                "target_text": target_text  # The model predicts the response.
            })
            counts[label] += 1
            total_added += 1
            if total_added >= casehold_count:
                break
        
        # Optionally, break early if all quotas are met.
        if all(counts[l] >= designated_counts[l] for l in labels):
            break

    return finetune_data

def get_billsum(billsum_count=7000, 
                 seed=1234,
                 split='train'):
    
    random.seed(seed)
    finetune_data = []

    if split == 'train':
        billsum_dataset = load_dataset('FiscalNote/billsum', trust_remote_code=True, split='train')
    elif split == 'validation':
        billsum_dataset = load_dataset('FiscalNote/billsum', trust_remote_code=True, split='test')
    else:
        raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'validation'.")

    count = 0
    for example in billsum_dataset:
        try:
            text_part = example.get('text', "").strip()
            title_part = example.get('title', "").strip()
            summary_part = example.get('summary', "").strip()

            if not text_part or not summary_part:
                continue  # Skip empty examples
        except KeyError:
            continue

        formatted_text = billsum_input_template.format(
            title=title_part,
            input_text=text_part,
        )
        formatted_target = billsum_target_template.format(
            summary=summary_part
        )

        finetune_data.append({
            "source": "BillSum",
            "input_text": formatted_text,
            "target_text": formatted_target
        })

        count += 1
        if count >= billsum_count:
            break

    return finetune_data

def get_contractnli(contractnli_count=7000,
                    entailment_prop=0.33, 
                    contradiction_prop=0.33,  
                    seed=1234,
                    split='train'):
    
    random.seed(seed)
    finetune_data = []
    contractNLI_dataset = load_dataset("kiddothe2b/contract-nli", "contractnli_a", split=split)
    neutral_prop = 1 - entailment_prop - contradiction_prop
    if neutral_prop < 0:
        raise ValueError("The sum of entailment_prop and contradiction_prop cannot exceed 1.")
    
    # Filter the data by each category.
    data_by_category = {
        "entailment": [entry for entry in contractNLI_dataset if entry["label"] == 1],
        "contradiction": [entry for entry in contractNLI_dataset if entry["label"] == 0],
        "neutral": [entry for entry in contractNLI_dataset if entry["label"] == 2]
    }
    # Determine the number of instances for each category.
    required_counts = {
        "entailment": int(round(contractnli_count * entailment_prop)),
        "contradiction": int(round(contractnli_count * contradiction_prop)),
        "neutral": int(round(contractnli_count * neutral_prop))
    }
    total_required = sum(required_counts.values())
    if total_required != contractnli_count:
        diff = contractnli_count - total_required
        # Adjust the category with the highest proportion.
        max_label = max(
            ("entailment", entailment_prop),
            ("contradiction", contradiction_prop),
            ("neutral", neutral_prop),
            key=lambda x: x[1]
        )[0]
        required_counts[max_label] += diff
    
    # Sample (or oversample) the specified number of entries from each category.
    sampled_entries = []
    for label, count in required_counts.items():
        available = data_by_category[label]
        if len(available) < count:
            print(f"Warning: Only {len(available)} available for label '{label}', needed {count}. Oversampling with replacement.")
            # Oversample with replacement using random.choices
            sampled = random.choices(available, k=count)
        else:
            sampled = random.sample(available, count)
        sampled_entries.extend(sampled)
    
    # Shuffle the combined entries.
    random.shuffle(sampled_entries)
    
    # Build the finetune data using the templates.
    for entry in sampled_entries:
        input_text = contractNLI_input_template.format(
            sentence1=entry['premise'],
            sentence2=entry['hypothesis']
        )
        target_text = contractNLI_target_template.format(
            summary={1: "entailment", 2: "neutral", 0: "contradiction"}.get(entry["label"], "unknown")
        )
        finetune_data.append({
            "source": "contractnli",
            "input_text": input_text,
            "target_text": target_text
        })
        
    return finetune_data

