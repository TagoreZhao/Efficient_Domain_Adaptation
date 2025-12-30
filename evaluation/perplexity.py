import torch
import torch.nn as nn
from tqdm import tqdm
from data.datasets import get_loaders

def eval_ppl(args, model, tokenizer, device=torch.device("cuda:0")):
    # Set dataset
    dataset = args.dataset
    print(f"evaluating on {dataset}")
    ppl_test = 0
    # Suppose `get_loaders` returns (trainloader, testloader)
    # Here we only need the testloader
    _, testloader = get_loaders(
        name=dataset, 
        seqlen=args.seqlen,
        seed=1234, 
        tokenizer=tokenizer
    )


    if args.dataset == "wikitext2":
        with torch.no_grad():
            ppl_test = eval_ppl_wikitext(
                model=model, 
                testenc=testloader, 
                seqlen=args.seqlen, 
                bs=1, 
                device=device
            )
    elif args.dataset == "harrison":
        with torch.no_grad():
            ppl_test = eval_ppl_Harrison(
                model=model, 
                testenc=testloader, 
                seqlen=args.seqlen, 
                bs=1, 
                device=device
            )
    elif args.dataset == "pplmultilegal":
        with torch.no_grad():
            ppl_test = eval_ppl_multilegalpile(
                model=model, 
                testenc=testloader, 
                seqlen=args.seqlen, 
                bs=1, 
                device=device
            )
        
    else:
        print("Perplexity over this dataset is not implemented yet")

    return 

def eval_ppl_wikitext(model, testenc, seqlen, bs=1, device=None):
    if seqlen <= 0:
        raise ValueError("seqlen must be a positive integer.")

    # testenc is presumably an encoded dataset, e.g. testenc.input_ids
    testenc = testenc.input_ids

    # Calculate how many samples we can form (seq chunks)
    nsamples = testenc.numel() // seqlen
    nlls = []
    print(f"nsamples {nsamples}")

    for i in range(0, nsamples, bs):
        if i % 50 == 0:
            print(f"sample {i}")

        j = min(i + bs, nsamples)

        # Slice out tokens for batch
        # shape: [1, seqlen] if bs=1
        inputs = testenc[:, (i * seqlen):(j * seqlen)].to(device)
        #print("The input has shape",inputs.shape)
        inputs = inputs.reshape(j - i, seqlen)
        #print("The input has shape",inputs.shape)
        # Forward pass with AMP

        lm_logits = model(inputs).logits

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), 
                        shift_labels.reshape(-1))

        # negative log likelihood for batch
        neg_log_likelihood = loss.float() * seqlen * (j - i)
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))

    # Clean up
    torch.cuda.empty_cache()
    return ppl.item()

def eval_ppl_Harrison(model, testenc, seqlen, bs=1, device="cuda"):
    """
    Evaluate perplexity on the Harrison textbook dataset.

    Parameters:
        model: Hugging Face model
            The language model to evaluate.
        testenc: Tensor
            Encoded test dataset.
        seqlen: int
            Sequence length for evaluation.
        bs: int, optional
            Batch size for evaluation. Default is 1.
        device: str, optional
            Device to run the model on ('cuda' or 'cpu').

    Returns:
        float: Perplexity of the evaluated dataset.
    """
    model.eval()  # Set model to evaluation mode
    input_ids = testenc.input_ids
    # nsamples = input_ids.size(1) // seqlen
    nsamples = 300
    nlls = []  # Negative log-likelihoods

    # Initialize tqdm progress bar
    with tqdm(total=nsamples, desc="Evaluating perplexity", unit="batch") as pbar:
        # Iterate over samples in batches
        for i in range(0, nsamples, bs):
            batch_input_ids = []
            for b in range(bs):
                idx = i + b
                if idx >= nsamples:
                    break
                start_idx = idx * seqlen
                end_idx = start_idx + seqlen
                batch_input_ids.append(input_ids[:, start_idx:end_idx])

            if not batch_input_ids:
                continue

            inputs = torch.cat(batch_input_ids, dim=0).to(device)

            # Forward pass
            with torch.no_grad():
                outputs = model(inputs)
                lm_logits = outputs.logits

            # Compute loss
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )

            neg_log_likelihood = loss.float()
            nlls.append(neg_log_likelihood)

            # Update progress bar
            pbar.update(bs)

    # Total negative log-likelihood and tokens
    total_nll = torch.stack(nlls).sum()
    total_tokens = nsamples * seqlen

    # Compute perplexity
    ppl = torch.exp(total_nll / total_tokens)

    return ppl.item()

def eval_ppl_multilegalpile(model, testenc, seqlen, bs=1, device="cuda"):
    """
    Evaluate perplexity on the first 300 samples in the multilegalpile US legislation dataset.

    Parameters:
        model: Hugging Face model
            The language model to evaluate.
        testenc: Tensor
            Encoded test dataset.
        seqlen: int
            Sequence length for evaluation.
        bs: int, optional
            Batch size for evaluation. Default is 1.
        device: str, optional
            Device to run the model on ('cuda' or 'cpu').

    Returns:
        float: Perplexity of the evaluated dataset.
    """
    model.eval()  # Set model to evaluation mode
    input_ids = testenc.input_ids
    # nsamples = input_ids.size(1) // seqlen
    nsamples = 300
    nlls = []  # Negative log-likelihoods

    # Iterate over samples in batches with progress tracking
    for i in tqdm(range(0, nsamples, bs), desc="Evaluating perplexity", unit="batch"):
        batch_input_ids = []
        for b in range(bs):
            idx = i + b
            if idx >= nsamples:
                break
            start_idx = idx * seqlen
            end_idx = start_idx + seqlen
            batch_input_ids.append(input_ids[:, start_idx:end_idx])

        if not batch_input_ids:
            continue

        inputs = torch.cat(batch_input_ids, dim=0).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(inputs)
            lm_logits = outputs.logits

        # Compute loss
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )

        neg_log_likelihood = loss.float()
        nlls.append(neg_log_likelihood)

    # Total negative log-likelihood and tokens
    total_nll = torch.stack(nlls).sum()
    total_tokens = nsamples * seqlen

    # Compute perplexity
    ppl = torch.exp(total_nll / total_tokens)

    return ppl.item()
