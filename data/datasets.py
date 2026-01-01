import json
import random
from turtle import pd
from datasets import load_dataset
import pandas as pd


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
    if 'wikitext2' in name:
        # wikitext2 does not have a padding argument
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if 'harrison' in name:
        # Harrison does not have a padding argument
        return get_Harrison(nsamples, seed, seqlen, tokenizer)
    if 'pplmultilegal' in name:
        return get_multilegalpile_ppl(nsamples, seed, seqlen, tokenizer)
    

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

def get_pubmedqa(pubmed_count=7000, positive=0.5, negative=0.3, seed=1234, file_path="data/downloaded/pubmedqa_maybe_output.json"):
    """
    Prepare fine-tuning data from PubMedQA dataset with properly formatted prompts.
    Sampling positive (yes), negative (no), and maybe examples according to specified proportions.
    """
    finetune_data = []
    random.seed(seed)

    # Load the PubMedQA dataset
    pubmed_data = load_dataset('qiaojin/PubMedQA', 'pqa_artificial', split='train')
    if len(pubmed_data) > pubmed_count:
        pubmed_data = pubmed_data.select(range(pubmed_count))

    # Group the data by final decision (yes, no, maybe)
    yes_data = []
    no_data = []
    with open(file_path, "r") as f:
        maybe_data = json.load(f)

    for row in pubmed_data:
        label = row['final_decision'].strip().lower()
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
        if len(data_list) >= desired_count:
            return random.sample(data_list, desired_count)
        else:
            return random.choices(data_list, k=desired_count)

    sampled_yes = sample_data(yes_data, desired_yes)
    sampled_no = sample_data(no_data, desired_no)
    sampled_maybe = sample_data(maybe_data, desired_maybe) if desired_maybe > 0 else []

    final_samples = sampled_yes + sampled_no + sampled_maybe
    random.shuffle(final_samples)

    # New input template matching evaluation
    pubmed_input_template = (
        "Below is an instruction that describes a task related to HealthCare, "
        "paired with detailed input from a scientific article.\n\n"
        "Instruction: As an expert doctor in clinical science and medical knowledge, "
        "can you tell me if the following statement is correct? Answer yes, no, or maybe.\n\n"
        "Input:\n"
        "Contexts:\n{contexts}\n\n"
        "Section Labels: {labels}\n\n"
        "MeSH Terms: {meshes}\n\n"
        "Question: {question}\n\n"
        "Response: The answer is"
    )

    pubmed_target_template = " {ground_truth}"

    for row in final_samples:
        contexts = "\n\n".join(row['context']['contexts']) if isinstance(row['context'], dict) else row.get('context', '')
        labels = ", ".join(row['context'].get('labels', [])) if isinstance(row['context'], dict) else ''
        meshes = ", ".join(row['context'].get('meshes', [])) if isinstance(row['context'], dict) else ''

        input_text = pubmed_input_template.format(
            contexts=contexts,
            labels=labels,
            meshes=meshes,
            question=row['question']
        )

        target_text = pubmed_target_template.format(
            ground_truth=row['final_decision'].strip().lower()
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
               mednli_file_path = 'data/downloaded/physionet.org/files/mednli/1.0.0/mli_train_v1.jsonl'):
    random.seed(seed)
    finetune_data = []
    mednli_input_template = (
        "Below is an instruction for analyzing clinical information.\n\n"
        "Instruction:\n"
        "You are a licensed physician tasked with determining the logical relationship between two clinical statements based on a patient's past medical history.\n"
        "Analyze carefully whether the second statement (hypothesis) can be inferred from the first statement (premise).\n"
        "Base your reasoning strictly on clinical knowledge, without making unwarranted assumptions.\n"
        "Your answer must be one of three options:\n"
        "- 'entailment' if the hypothesis must be true given the premise,\n"
        "- 'contradiction' if the hypothesis must be false given the premise,\n"
        "- 'neutral' if the hypothesis could be true or false without certainty.\n"
        "Fully consider all three possibilities and reason carefully before giving your final answer.\n\n"
        "Input:\nPremise: '{sentence1}'\nHypothesis: '{sentence2}'\n\n"
        "Final Answer:"
    )
    mednli_target_template = " Their relationship is {gold_label}"
    
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

def get_hqs(hqs_count = 1000, data_dir = 'data/downloaded/MeQSum_ACL2019_BenAbacha_Demner-Fushman.xlsx'):
    
    prompt_template = (
            "Below is an instruction that describes a task related to HealthCare, paired with further context. "
            "Write a response that appropriately completes the request.\n\n"
            "Instruction: Summarize the following consumer health question by generating a condensed version that retains all critical information necessary to find correct and complete answers. "
            "Focus on preserving key entities (e.g., conditions, treatments, tests) and the main intent of the question, while omitting unnecessary peripheral details. "
            "Write the summary fluently in natural language and avoid simply shortening without ensuring information completeness.\n\n"
            "Please provide the shortened version directly.\n\n"
            "Input: {input_question}\n\nShortened Question:"
    )
    
    hqs_target_template = " {sum}"
    finetune_data = []
    # Load the MeQSum dataset from Excel.
    hqs_df = pd.read_excel(data_dir)
    hqs_df = hqs_df.head(hqs_count)
    
    for _, row in hqs_df.iterrows():
        input_text = prompt_template.format(input_question=row['CHQ'])
        target_text = hqs_target_template.format(sum=row['Summary'])  # Adjust column name as needed.
        finetune_data.append({
            "source": "MeQSum",
            "input_text": input_text,
            "target_text": target_text
        })

    return finetune_data


def get_casehold(casehold_count=7000, 
                 choice_prop=[1/5, 1/5, 1/5, 1/5, 1/5],
                 seed=1234):
    random.seed(seed)
    
    finetune_data = []
    casehold_dataset = load_dataset('casehold/casehold', split='train', trust_remote_code=True)
    
    casehold_template = (
        "Below is an instruction that describes a task related to making legal decisions based on a citing prompt. "
        "Please read the citing prompt, and decide which holding statement best corresponds to it.\n\n"
        "Instruction:\n"
        "You are given a citing passage from a legal decision and five potential holding statements.\n"
        "Your task is to carefully read the citing prompt and select the holding statement that best corresponds to it.\n"
        "Focus on the precise legal principle, factual framing, and implications described.\n"
        "Base your decision purely on the logical match between the citing prompt and the holding, avoiding irrelevant or merely similar content.\n\n"
        "Your answer should be 0, 1, 2, 3, or 4.\n\n"
        "Input: citing_prompt: {citing_prompt}, 0: {holding_0}, 1: {holding_1}, 2: {holding_2}, 3: {holding_3}, 4: {holding_4}\n\n"
        "Response: The answer is"
    )
    casehold_target_template = " {label}"
    
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
            formatted_text = casehold_template.format(
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
                 seed=1234):
    
    random.seed(seed)
    finetune_data = []
    billsum_dataset = load_dataset('FiscalNote/billsum', trust_remote_code=True, split='train')
    billsum_template = (
        "Below is an instruction for summarizing a legislative bill.\n\n"
        "Instruction:\n"
        "Summarize the following legislative text by clearly identifying and explaining the *major actions, purposes, and effects* of the bill. "
        "Focus on what the bill aims to achieve rather than technical details or administrative changes. "
        "Paraphrase the information in a simple and accessible way as if writing for policymakers and the public. "
        "Avoid copying long passages or citing subsection numbers unless necessary for understanding.\n\n"
        "Input:\n"
        "Bill Title: {title}\n\n"
        "{input_text}\n\n"
        "Response:"
    )
    billsum_target_template = "{summary}"
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

        formatted_text = billsum_template.format(
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
                    seed=1234):
    
    random.seed(seed)
    finetune_data = []
    contractNLI_dataset = load_dataset("kiddothe2b/contract-nli", "contractnli_a", split="train")
    # contractNLI_template = (
    #     "Below is an instruction describing a task related to legal contract analysis.\n\n"
    #     "Instruction:\n"
    #     "You are given a pair consisting of a Premise and a Hypothesis, both drawn from a formal legal contract.\n"
    #     "Carefully read the Premise and determine whether the Hypothesis:\n"
    #     " - (Entailment) Must be true if the Premise is true,\n"
    #     " - (Contradiction) Is directly refuted by the Premise,\n"
    #     " - (Neutral) Is neither clearly entailed nor contradicted by the Premise.\n\n"
    #     "You must think carefully and reason step-by-step, considering the precise meaning of the contract language.\n"
    #     "Your final answer must be one of exactly 'entailment', 'contradiction', or 'neutral'.\n\n"
    #     "Premise: {sentence1}\n"
    #     "Hypothesis: {sentence2}\n\n"
    #     "Response:\n"
    #     "Final Answer:"
    # )
    contractNLI_template = (
        "Below is an instruction that describes a task related to Contracts, paired with an input that provides context. "
        "Write a detailed response that includes your reasoning process, then conclude with your final answer.\n\n"
        "Instruction: Carefully analyze the relationship between the Contract Premise and Hypothesis. "
        "Your final answer must be one of ‘entailment’, ‘contradiction’, or ‘neutral’. "
        "Ensure that you fully consider all three relationships.\n\n"
        "Input:\nPremise: '{sentence1}'\nHypothesis: '{sentence2}'\n\n"
        "Final Answer: Their relationship is "
    )
    contractNLI_target_template = "{summary}"
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
        input_text = contractNLI_template.format(
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
