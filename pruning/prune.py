import torch
import torch.nn as nn
from .WrappedGPT import WrappedGPT
from data.datasets import get_loaders
from .utils import get_layers, find_layers, layer_forward, return_given_alpha, prepare_calibration_input


@torch.no_grad()
def prune_wanda(
    sparsity_ratio,
    nsamples,
    seed,
    seqlen,
    model,
    tokenizer,
    dataset_name="c4",
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0,
    use_variant=False,
):
    """
    Wanda pruning adapted for Qwen3 series.

    Requirements:
      - get_loaders(...) must exist and return a list or DataLoader of (input_ids, target_ids)
      - prepare_calibration_input(...) must exist and return (inps, outs, attention_mask, position_ids),
        where inps/outs are hidden-states entering layer 0: [nsamples, seqlen, hidden]

    Key Qwen3 fix:
      - When calling a single decoder layer directly, we must pass `position_embeddings`
        (cos,sin) computed from model.model.rotary_emb(hidden_states, position_ids).
    """
    # Cache config
    use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False

    model.eval()
    if device is None:
        device = next(model.parameters()).device

    print("loading calibration data")
    dataloader, _ = get_loaders(
        dataset_name,
        nsamples=nsamples,
        seed=seed,
        seqlen=seqlen,
        tokenizer=tokenizer,
    )
    print("dataset loading complete")

    # Collect calibration hidden-states that enter layer 0
    inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device=device)

    layers = get_layers(model)

    # Iterate blocks
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer, layers=(nn.Linear,))

        # Handle model parallel device map (optional)
        if hasattr(model, "hf_device_map") and f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = attention_mask.to(dev) if attention_mask is not None else None
            position_ids = position_ids.to(dev) if position_ids is not None else None
        else:
            dev = inps.device

        # Wrap linears to collect activation stats
        wrapped_layers = {name: WrappedGPT(mod, layer_id=i, layer_name=name) for name, mod in subset.items()}

        def add_batch(name):
            def _hook(mod, inp, out):
                # inp is a tuple; inp[0] is the tensor input to this Linear
                wrapped_layers[name].add_batch(inp[0].detach(), out.detach())
            return _hook

        handles = []
        for name, mod in subset.items():
            handles.append(mod.register_forward_hook(add_batch(name)))

        # Run layer forward nsamples times to populate wrapped_layers[name].scaler_row
        # IMPORTANT: Use Qwen3-correct layer forward (provides position_embeddings)
        for j in range(nsamples):
            hs = inps[j].unsqueeze(0)  # [1, seqlen, hidden]
            outs[j] = layer_forward(
                model=model,
                layer=layer,
                hidden_states=hs,
                attention_mask=attention_mask,
                position_ids=position_ids,
            ).squeeze(0)

        for h in handles:
            h.remove()

        # Compute Wanda metric and prune
        for name, mod in subset.items():
            print(f"pruning layer {i} name {name}")

            # Wanda metric: |W| * sqrt(activation_row_scaler)
            scaler = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1))).to(mod.weight.device)
            W_metric = torch.abs(mod.weight.data) * scaler

            W_mask = torch.zeros_like(W_metric, dtype=torch.bool)

            if prune_n != 0:
                # structured n:m sparsity per row blocks along columns
                assert prune_m > 0 and prune_n <= prune_m
                for col in range(0, W_metric.shape[1], prune_m):
                    block = W_metric[:, col:col + prune_m].float()
                    # pick n smallest in each row of the block
                    idx = torch.topk(block, prune_n, dim=1, largest=False).indices
                    W_mask.scatter_(1, col + idx, True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if use_variant:
                    tmp_metric = torch.cumsum(sort_res.values, dim=1) if hasattr(sort_res, "values") else torch.cumsum(sort_res[0], dim=1)
                    sorted_vals = sort_res.values if hasattr(sort_res, "values") else sort_res[0]
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0.0, 0.8]

                    # return_given_alpha expects sort_res as (values, indices)
                    if hasattr(sort_res, "values"):
                        sort_tuple = (sort_res.values, sort_res.indices)
                    else:
                        sort_tuple = sort_res

                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_tuple, W_metric, tmp_metric, sum_before)

                    # binary search alpha to match target sparsity
                    while (torch.abs(cur_sparsity - sparsity_ratio) > 0.001) and ((alpha_hist[1] - alpha_hist[0]) >= 0.001):
                        if cur_sparsity > sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha
                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_tuple, W_metric, tmp_metric, sum_before)

                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured: prune fixed fraction by rank (per row)
                    idx = sort_res.indices if hasattr(sort_res, "indices") else sort_res[1]
                    k = int(W_metric.shape[1] * sparsity_ratio)
                    indices = idx[:, :k]
                    W_mask.scatter_(1, indices, True)

            # Apply pruning
            mod.weight.data[W_mask] = 0.0

        # Recompute outs after pruning for next-layer input propagation
        for j in range(nsamples):
            hs = inps[j].unsqueeze(0)
            outs[j] = layer_forward(
                model=model,
                layer=layer,
                hidden_states=hs,
                attention_mask=attention_mask,
                position_ids=position_ids,
            ).squeeze(0)

        # Swap buffers for next layer
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    return model