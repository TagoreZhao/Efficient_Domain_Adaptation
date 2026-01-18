import torch
import torch.nn as nn
from .WrappedGPT import WrappedGPT
from data.datasets import get_loaders
from .utils import (get_layers, 
                    find_layers, 
                    layer_forward, 
                    return_given_alpha, 
                    prepare_calibration_input, 
                    merge_lora_into_base, 
                    reset_lora,
                    get_comp_norm)
from peft.tuners.lora import Linear

@torch.no_grad()
def foresight_prune(model,
                dataloader, 
                prune_ratio,
                mask_lr,
                nsamples=128,
                device=None,
                merge_lora=True,
                PBS=True):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    num_q_heads = model.config.num_attention_heads
    head_dim = model.config.head_dim
    num_kv_groups = model.config.num_key_value_heads

    model.eval()
    if device is None:
        device = next(model.parameters()).device

    print("preparing calibration input")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader)
    print("calibration input prepared")

    layers = get_layers(model)
    
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer, layers=(Linear,))

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
        wrapped_layers = {name: WrappedGPT(mod.base_layer, layer_id=i, layer_name=name) for name, mod in subset.items()}

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

        #please edit the following part to compute correct importance score and add it to score, set mask based on the importance score
        for name in subset:
            # Access the relevant tensors for each module.
            print(f"Pruning layer {i}, param {name}")
            # print(torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9)
            weight       = subset[name].base_layer.weight
            lora_A       = subset[name].lora_A.default.weight
            lora_B       = subset[name].lora_B.default.weight
            if not hasattr(subset[name], "score") or subset[name].score is None:
                print("Initializing score tensor")
                subset[name].score = torch.zeros_like(weight, device=torch.device("cpu"))
            score_cpu = subset[name].score
            # X_hat is computed from the activation statistics (assumed to be feature norm)
            X_hat = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1))).to(mod.weight.device)
            # Save the original devices for score and mask
            orig_score_device = score_cpu.device

            # Move score and mask to the current computation device (assume weight's device)
            prev = score_cpu.to(device=weight.device, dtype=weight.dtype, non_blocking=True)

            if name == "self_attn.q_proj":
                effective_weight = weight + torch.matmul(lora_B, lora_A)
                effective_weight_Q = effective_weight.view(num_q_heads, head_dim, -1)
                abs_Q = torch.abs(effective_weight_Q)
                comp_layer = subset["self_attn.k_proj"]
                comp_weight = comp_layer.base_layer.weight + torch.matmul(
                    comp_layer.lora_B.default.weight,
                    comp_layer.lora_A.default.weight,
                )
                effective_weight_K = comp_weight.view(num_kv_groups, head_dim, -1)
                kv_group_norms = torch.norm(effective_weight_K, p=2, dim=1)
                group_size = num_q_heads // num_kv_groups
                kv_group_idx = torch.arange(num_q_heads, device=abs_Q.device) // group_size  # [num_q_heads]
                kv_norms_expanded = kv_group_norms[kv_group_idx]
                X_hat = X_hat.view(1, 1, -1).to(abs_Q.device)
                importance = torch.mul(abs_Q, kv_norms_expanded.unsqueeze(1))
                importance = torch.mul(importance, X_hat)
                importance = importance.view(-1, importance.shape[-1])

            if name == "self_attn.k_proj":
                effective_weight = weight + torch.matmul(lora_B, lora_A)
                k_weight_grouped = effective_weight.view(num_kv_groups, head_dim, -1)
                k_weight_abs = torch.abs(k_weight_grouped)
                comp_layer = subset["self_attn.q_proj"]
                q_weight = comp_layer.base_layer.weight + torch.matmul(
                    comp_layer.lora_B.default.weight,
                    comp_layer.lora_A.default.weight,
                )
                group_size = num_q_heads // num_kv_groups
                q_weight_grouped = q_weight.view(num_kv_groups, group_size, head_dim, -1)
                q_head_norm = torch.norm(q_weight_grouped, p=2, dim=2)
                q_group_norm = torch.sqrt(torch.mean(torch.mul(q_head_norm, q_head_norm), dim=1, keepdim=True))
                X_hat = X_hat.view(1, 1, -1).to(k_weight_abs.device)
                importance_grouped = torch.mul(k_weight_abs, q_group_norm)
                importance_grouped = torch.mul(importance_grouped, X_hat)
                importance = importance_grouped.view(-1, importance_grouped.shape[-1])

            elif name == "self_attn.v_proj":
                # Compute effective v_proj weight: shape [512, 2048] → 8 heads × 64 rows
                effective_weight = weight + torch.matmul(lora_B, lora_A)
                out_features, in_features = effective_weight.shape
                g = num_q_heads // num_kv_groups
                Wv_h = effective_weight.view(num_kv_groups, head_dim, in_features)  # [h_kv, d_h, in_features]
                comp_layer = subset["self_attn.o_proj"]
                comp_weight = comp_layer.base_layer.weight + torch.matmul(
                                    comp_layer.lora_B.default.weight,
                                    comp_layer.lora_A.default.weight)
                
                Wo_blocks = comp_weight.view(comp_weight.shape[0], num_q_heads, head_dim)
                Wo_grouped = Wo_blocks.view(comp_weight.shape[0], num_kv_groups, g, head_dim)
                tmp = torch.norm(Wo_grouped, p=2, dim=0)          # norm over out_dim -> [h_kv, g, d_h]
                comp_kv = torch.sqrt(tmp.pow(2).mean(dim=1))       # RMS over g -> [h_kv, d_h]
                importance_grouped = torch.abs(Wv_h) * comp_kv.unsqueeze(-1) * X_hat  # shape: (8, 64, 2048)
                importance = importance_grouped.view(out_features, in_features)

            elif name == "self_attn.o_proj":
                # Compute o_proj effective weight: (2048, 2048)
                effective_weight = weight + torch.matmul(lora_B, lora_A)

                norm_gate = get_comp_norm(comp = subset["mlp.gate_proj"], dim=0)
                norm_up   = get_comp_norm(comp = subset["mlp.up_proj"], dim=0)
                norm = (norm_gate + norm_up) / 2
                norm = norm.reshape(-1, 1)
                # Compute importance as per:
                # importance_{ij} = |(W1+B1A1)_{ij}| * ||X_{:,i}||_2 * (norm_gate + norm_up)_j
                importance = torch.abs(effective_weight) * X_hat * norm
            elif name == "mlp.gate_proj":
                effective_weight = weight + torch.matmul(lora_B, lora_A)
                comp_norm = get_comp_norm(comp = subset["mlp.down_proj"], dim=1)
                importance = torch.abs(effective_weight) * comp_norm * X_hat  # shape: (8192, 2048)
            elif name == "mlp.up_proj":
                # effective_weight: shape [8192, 2048]
                effective_weight = weight + torch.matmul(lora_B, lora_A)
                comp_norm = get_comp_norm(comp = subset["mlp.down_proj"], dim=1)
                importance = torch.abs(effective_weight) * comp_norm * X_hat
            elif name == "mlp.down_proj":
                effective_weight = weight + torch.matmul(lora_B, lora_A)
                importance = torch.abs(effective_weight) * X_hat
            else:
                print("This weight is not being recognized " + name)
                comp_weight = None

            if torch.count_nonzero(prev).item() > 0:
                # Moving average: new_score = mask_lr * current + (1 - mask_lr) * previous
                print("we use moving average")
                importance = mask_lr * importance + (1 - mask_lr) * prev

            # Update the stored score.
            score_cpu.copy_(importance.detach().to(orig_score_device))
            # merge the lora adpater, apply the mask, and reinitialized the lora adapter
            
            # Rowwise Selection
            W_mask = torch.zeros_like(importance, dtype=torch.bool)
            num_to_prune = int(importance.shape[1] * prune_ratio)
            sort_res = torch.sort(importance, dim=-1, stable=True)
            indices = sort_res[1][:, :num_to_prune]
            rows = torch.arange(importance.shape[0]).unsqueeze(-1).to(importance.device)
            W_mask[rows, indices] = True

            if PBS:
                # Perform partial brain surgeon recovery
                print("Performing PBS recovery")
                lora_B_new = partial_brain_surgeon(
                    X_hat=X_hat,
                    M=~W_mask,
                    B=lora_B,
                    A=lora_A,
                    W=weight,
                    reg=1e-8,
                    keep_weight=1.0,
                    max_pruned_per_row=None,
                    eps=1e-12,
                    inplace=True,
                )

            if merge_lora:
                merge_lora_into_base(subset[name], adapter="default")
                reset_lora(subset[name], adapter="default")
            
            weight.masked_fill_(W_mask, 0.0)


        for j in range(min(128, inps.shape[0])):
            with torch.no_grad():
                pos_emb = model.base_model.model.model.rotary_emb(x = inps[j].unsqueeze(0),position_ids = position_ids)
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=pos_emb
                )[0]
        inps, outs = outs, inps

    # Restore original cache setting
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

@torch.no_grad()
def prune_wanda(
    sparsity_ratio,
    nsamples,
    seed,
    seqlen,
    model,
    tokenizer,
    dataset_name="c4",
    device=None,
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

@torch.no_grad()
def partial_brain_surgeon(
    X_hat,
    M,
    B,
    A,
    W,
    reg=1e-8,
    keep_weight=1.0,
    max_pruned_per_row=None,
    eps=1e-12,
    inplace=False,
    xhat_is_squared=True,
):
    """
    Adapter recovery under a fixed mask M (single-layer, closed-form), updating B only (A fixed).

    This version matches the common transformer Linear shapes you provided:
        X_hat:   [1, 1, n]  (here n=1024)  -> interpreted as per-input-feature energy
        W:       [m, n]     (here m=2048, n=1024)
        B:       [m, r]     (here r=8)
        A:       [r, n]

    IMPORTANT: With X_hat shaped like (.., n), we treat it as *column weights*:
        g_j ≈ (X^T X)_{jj} = ||X_{:,j}||_2^2,  j=1..n

    Row-wise objective (for each output row i):
        min_{Δb_i}
            || (U_{i,S_i} + Δb_i A_{:,S_i}) diag(sqrt(g_{S_i})) ||_2^2
          + keep_weight * || (Δb_i A_{:,K_i}) diag(sqrt(g_{K_i})) ||_2^2
          + reg * ||Δb_i||_2^2

    where U = W + BA (current effective weight),
          S_i are pruned columns in row i, K_i are kept columns in row i.

    Closed form uses weighted Grams:
        A D A^T and A_{S} D_S A_{S}^T, with D=diag(g).

    Args:
        X_hat: Tensor with last dim = n (preferred), e.g. [1,1,n] or [n].
               If last dim == m, we fall back to row-weighting (older behavior).
        M:     Binary mask [m,n], 0=pruned, 1=kept.
        B,A,W: As above.
        reg:   Ridge (Tikhonov) coefficient.
        keep_weight: Strength of "do-no-harm on kept coordinates" penalty.
        max_pruned_per_row: Optional cap on |S_i| via largest |U_{i,S_i}|.
        eps:   Numerical floor for g.
        inplace: Update B in-place if True.
        xhat_is_squared: If True, X_hat already stores ||X_{:,j}||^2.
                         If False, we will square it to get g.

    Returns:
        Updated B with same shape/device/dtype as input B.
    """
    # dtype/device harmonization
    if B.dtype != A.dtype:
        B = B.to(A.dtype)
    device = A.device
    dtype = A.dtype

    # shapes
    m, r = B.shape
    rA, n = A.shape
    assert rA == r, f"A has shape {A.shape}, but B has r={r}."
    assert W.shape == (m, n), f"W shape {W.shape} must be (m,n)=({m},{n})."
    assert M.shape == (m, n), f"M shape {M.shape} must be (m,n)=({m},{n})."

    # move mask once to GPU
    M_dev = M.to(device=device)

    # clone or inplace
    B_new = B if inplace else B.clone()

    # current effective weight
    U = W + B_new @ A  # (m, n)

    # parse X_hat into a 1D vector
    g_raw = X_hat
    if not torch.is_tensor(g_raw):
        g_raw = torch.tensor(g_raw)
    g_raw = g_raw.to(device=device, dtype=dtype).reshape(-1)

    # Two supported interpretations:
    g = g_raw
    if not xhat_is_squared:
        g = g * g
    g = torch.clamp(g, min=eps)  # (n,)

    # Precompute A D A^T with D=diag(g):  ADAT = A diag(g) A^T
    # Efficiently: (A * g) @ A^T where g broadcasts over columns
    ADAT = (A * g.unsqueeze(0)) @ A.t()  # (r, r)

    I_r = torch.eye(r, device=device, dtype=dtype)
    reg_t = torch.tensor(reg, device=device, dtype=dtype)

    for i in range(m):
        # pruned columns in this row
        S_i = (M_dev[i] == 0).nonzero(as_tuple=True)[0]
        if S_i.numel() == 0:
            continue

        # optionally cap |S_i|
        if max_pruned_per_row is not None and S_i.numel() > max_pruned_per_row:
            x_pruned = U[i, S_i].abs()
            topk = torch.topk(x_pruned, k=max_pruned_per_row, largest=True).indices
            S_i = S_i[topk]

        # gather sub-vectors/matrices
        g_S = g[S_i]                 # (k,)
        A_S = A[:, S_i]              # (r, k)
        x_S = U[i, S_i]              # (k,)

        # Weighted Gram on pruned set: A_S diag(g_S) A_S^T
        # via (A_S * g_S) @ A_S^T
        ApDpApT = (A_S * g_S.unsqueeze(0)) @ A_S.t()  # (r, r)

        # Using identity: A_K D_K A_K^T = ADAT - ApDpApT
        # Gram = ApDpApT + keep_weight*(ADAT - ApDpApT) + reg I
        #      = keep_weight*ADAT + (1-keep_weight)*ApDpApT + reg I
        Gram = keep_weight * ADAT + (1.0 - keep_weight) * ApDpApT + reg_t * I_r  # (r, r)

        # rhs = x_S diag(g_S) A_S^T  == (x_S * g_S) @ A_S^T
        rhs = (x_S * g_S) @ A_S.t()  # (r,)

        # Solve Gram * z = rhs^T  => delta_b = -z^T (row vector)
        z = torch.linalg.solve(Gram, rhs.unsqueeze(1)).squeeze(1)  # (r,)
        B_new[i] = B_new[i] - z

    return B_new
