import torch
import math
import torch.nn as nn
from tqdm import tqdm
from data.datasets import get_loaders

def eval_ppl(dataset, seqlen, model, tokenizer, device=torch.device("cuda:0"), seed = 1234):
    
    if dataset not in ['wikitext2', 'harrison', 'pplmultilegal']:
        raise ValueError(f"Unsupported dataset: {dataset}")

    print(f"evaluating on {dataset}")
    ppl_test = 0
    # Suppose `get_loaders` returns (trainloader, testloader)
    # Here we only need the testloader
    _, testloader = get_loaders(
        name=dataset, 
        seqlen=seqlen,
        seed=seed, 
        tokenizer=tokenizer
    )

    ppl_test = eval_ppl_help(
        model=model, 
        testenc=testloader, 
        seqlen=seqlen, 
        device=device,
        max_blocks=300
    )

    return ppl_test

def eval_ppl_help(model, testenc, seqlen, bs=1, device=None, max_blocks=None, verbose_every=50):
    """
    Perplexity over a long token stream using non-overlapping blocks.
    Computes: ppl = exp(total_nll / total_predicted_tokens)
    where total_predicted_tokens = nblocks * (seqlen - 1).
    """
    if seqlen <= 1:
        raise ValueError("seqlen must be > 1.")

    model.eval()
    if device is None:
        device = next(model.parameters()).device

    input_ids = testenc.input_ids  # tensor-like; must support slicing [:, start:end]
    # Ensure shape [1, N] if a raw tensor is provided
    if isinstance(input_ids, torch.Tensor) and input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    # Determine total length N
    if isinstance(input_ids, torch.Tensor):
        N = input_ids.size(1)
    else:
        # proxy: provides .shape
        N = input_ids.shape[1]

    nblocks = N // seqlen
    if max_blocks is not None:
        nblocks = min(nblocks, int(max_blocks))

    if nblocks <= 0:
        raise ValueError(f"Not enough tokens ({N}) for seqlen={seqlen}.")

    loss_fct = nn.CrossEntropyLoss(reduction="sum")

    total_nll = 0.0
    total_pred_tokens = 0

    with torch.no_grad():
        for b0 in range(0, nblocks, bs):
            b1 = min(b0 + bs, nblocks)

            if verbose_every and (b0 % (verbose_every * bs) == 0):
                print(f"block {b0}/{nblocks}")

            batch = []
            for bi in range(b0, b1):
                start = bi * seqlen
                end = start + seqlen
                batch.append(input_ids[:, start:end])  # each is [1, seqlen] tensor

            inputs = torch.cat(batch, dim=0).to(device)  # [B, seqlen]
            logits = model(inputs).logits                # [B, seqlen, vocab]

            shift_logits = logits[:, :-1, :].contiguous()  # [B, seqlen-1, vocab]
            shift_labels = inputs[:, 1:].contiguous()      # [B, seqlen-1]

            nll = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            total_nll += float(nll.item())
            total_pred_tokens += (b1 - b0) * (seqlen - 1)

    ppl = math.exp(total_nll / total_pred_tokens)
    return ppl
