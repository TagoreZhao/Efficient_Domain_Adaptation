import torch
import torch.nn as nn
from tqdm import tqdm
from data.datasets import get_loaders

def eval_ppl(dataset, seqlen, model, tokenizer, device=torch.device("cuda:0")):
    # Set dataset
    dataset = dataset
    print(f"evaluating on {dataset}")
    ppl_test = 0
    # Suppose `get_loaders` returns (trainloader, testloader)
    # Here we only need the testloader
    _, testloader = get_loaders(
        name=dataset, 
        seqlen=seqlen,
        seed=1234, 
        tokenizer=tokenizer
    )


    if dataset == "wikitext2":
        with torch.no_grad():
            ppl_test = eval_ppl_wikitext(
                model=model, 
                testenc=testloader, 
                seqlen=seqlen, 
                device=device
            )
    elif dataset == "harrison":
        with torch.no_grad():
            ppl_test = eval_ppl_Harrison(
                model=model, 
                testenc=testloader, 
                seqlen=seqlen, 
                bs=1, 
                device=device
            )
    elif dataset == "pplmultilegal":
        with torch.no_grad():
            ppl_test = eval_ppl_multilegalpile(
                model=model, 
                testenc=testloader, 
                seqlen=seqlen, 
                bs=1, 
                device=device
            )
        
    else:
        print("Perplexity over this dataset is not implemented yet")

    return ppl_test

import torch
import torch.nn as nn

def eval_ppl_wikitext(model, testenc, seqlen, device=None):
    """
    Evaluate perplexity on a long token stream using non-overlapping blocks (no stride),
    processing one block at a time (no batch processing).

    PPL = exp(total_NLL / total_predicted_tokens), where each block contributes (seqlen-1)
    predicted tokens because of next-token shifting.
    """
    if seqlen <= 1:
        raise ValueError("seqlen must be > 1 (need at least 2 tokens to predict next-token).")

    model.eval()
    if device is None:
        device = next(model.parameters()).device

    input_ids = testenc.input_ids
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)  # [1, N]

    nsamples = input_ids.numel() // seqlen
    print(f"nsamples {nsamples}")

    total_nll = 0.0
    total_pred_tokens = 0

    loss_fct = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for i in range(nsamples):
            if i % 50 == 0:
                print(f"sample {i}")

            # One block: [1, seqlen]
            inputs = input_ids[:, i * seqlen : (i + 1) * seqlen].to(device)

            logits = model(inputs).logits  # [1, seqlen, vocab]

            shift_logits = logits[:, :-1, :].contiguous()   # [1, seqlen-1, vocab]
            shift_labels = inputs[:, 1:].contiguous()       # [1, seqlen-1]

            nll = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            total_nll += nll.item()
            total_pred_tokens += (seqlen - 1)

    ppl = torch.exp(torch.tensor(total_nll / total_pred_tokens))

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
