import random
from datasets import load_dataset


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
