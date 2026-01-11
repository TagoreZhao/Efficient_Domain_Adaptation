import torch
import torch.nn as nn

def find_layers(module, layers=(nn.Linear,), name=""):
    """
    Recursively find the layers of certain type(s) in a module.
    Returns: dict[name -> module]
    """
    if isinstance(module, layers):
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        child_name = name + "." + name1 if name != "" else name1
        res.update(find_layers(child, layers=layers, name=child_name))
    return res

def get_layers(model: nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError("Expected Qwen3-style `model.model.layers`.")

def make_position_ids(position_ids, seqlen, bsz, device):
    """
    Ensure position_ids is shape [bsz, seqlen] on device.
    If None, create 0..seqlen-1.
    """
    if position_ids is None:
        return torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)  # [bsz, seqlen]
    position_ids = position_ids.to(device)
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    if position_ids.shape[0] == 1 and bsz > 1:
        position_ids = position_ids.expand(bsz, -1)
    return position_ids

def compute_position_embeddings(model, hidden_states, position_ids):
    """
    Qwen3 expects `position_embeddings` as a tuple (cos, sin), produced by model.model.rotary_emb.
    """
    if not (hasattr(model, "model") and hasattr(model.model, "rotary_emb")):
        raise AttributeError("Expected Qwen3-style `model.model.rotary_emb` to compute RoPE embeddings.")
    return model.model.rotary_emb(hidden_states, position_ids)  # (cos, sin)

def layer_forward(model, layer, hidden_states, attention_mask=None, position_ids=None):
    """
    Forward a single Qwen3 decoder layer correctly by providing position_embeddings.
    hidden_states: [bsz, seqlen, hidden]
    Returns: hidden_states_out [bsz, seqlen, hidden]
    """
    bsz, seqlen, _ = hidden_states.shape
    dev = hidden_states.device

    pos_ids = make_position_ids(position_ids, seqlen, bsz, dev)
    pos_emb = compute_position_embeddings(model, hidden_states, pos_ids)

    out = layer(
        hidden_states,
        attention_mask=attention_mask,          # may be None; OK if your sequences are unpadded
        position_ids=pos_ids,
        position_embeddings=pos_emb,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
    )
    # HF layers typically return a tuple; first element is hidden_states
    return out[0] if isinstance(out, (tuple, list)) else out

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def _get_decoder_layers(model: nn.Module):
    """
    Qwen3ForCausalLM typically has model.model.layers (Qwen3Model.layers).
    This helper makes the access explicit and fail-fast with a clear error.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError("Could not find decoder layers. Expected `model.model.layers` for Qwen3.")


class _StopForward(Exception):
    """Internal exception used to early-exit the forward pass after capturing activations."""
    pass


@torch.no_grad()
def prepare_calibration_input(model, dataloader, device=None, max_samples=None):
    """
    Capture the hidden-states entering the first decoder layer for a set of calibration samples,
    without replacing the layer module (uses a forward pre-hook instead).

    Returns:
        inps:          [n_to_capture, seqlen, hidden]
        outs:          same shape as inps (zeros)
        attention_mask: last seen attention_mask passed into layer 0 (may be None)
        position_ids:   last seen position_ids passed into layer 0 (may be None)
    """
    if device is None:
        device = next(model.parameters()).device

    model_was_training = model.training
    model.eval()

    # Disable kv-cache during calibration
    use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False

    layers = _get_decoder_layers(model)

    # Helper to normalize batch -> input_ids (batch, seqlen)
    def _get_input_ids(batch):
        # your dataset returns (inp, tar); DataLoader returns same but with batch dim
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, seqlen]
        return x

    # Grab one batch to infer seqlen
    it = iter(dataloader)
    first = next(it)
    first_ids = _get_input_ids(first)
    seqlen = first_ids.shape[1]

    # Determine how many samples we will actually capture
    total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
    n_to_capture = total_batches if total_batches is not None else 0
    if max_samples is not None:
        n_to_capture = min(n_to_capture, max_samples) if n_to_capture else max_samples

    # If we can't know length, we require max_samples
    if n_to_capture == 0:
        if max_samples is None:
            raise ValueError("Dataloader has no __len__; pass max_samples.")
        n_to_capture = max_samples

    dtype = next(model.parameters()).dtype
    hidden = model.config.hidden_size

    inps = torch.zeros((n_to_capture, seqlen, hidden), dtype=dtype, device=device)
    outs = torch.zeros_like(inps)

    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    # Pre-hook on the first decoder layer: capture inputs to that layer
    def _pre_hook(module, args, kwargs):
        """
        args[0] should be hidden_states entering decoder layer 0: [bsz, seqlen, hidden]
        kwargs may include attention_mask, position_ids, etc.
        """
        if not args:
            raise RuntimeError("Unexpected: decoder layer received no positional args (hidden_states missing).")

        hs = args[0].to(device)  # [bsz, seqlen, hidden]
        bsz = hs.shape[0]

        # Capture sample-by-sample (works for batch_size=1; supports >1 until buffer fills)
        for b in range(bsz):
            if cache["i"] >= inps.shape[0]:
                break
            inps[cache["i"]] = hs[b]
            cache["i"] += 1

        cache["attention_mask"] = kwargs.get("attention_mask", cache["attention_mask"])
        cache["position_ids"] = kwargs.get("position_ids", cache["position_ids"])

        # Early stop once we have enough samples; otherwise allow forward to continue
        if cache["i"] >= inps.shape[0]:
            raise _StopForward()

        # If you want to always stop immediately after layer-0 capture (faster but needs
        # one model() call per sample/batch), uncomment the next line:
        # raise _StopForward()

    hook_handle = layers[0].register_forward_pre_hook(_pre_hook, with_kwargs=True)

    def _run_batch(batch):
        input_ids = _get_input_ids(batch).to(device)
        # run until we either fill inps (hook raises _StopForward) or forward completes
        model(input_ids=input_ids, use_cache=False)

    try:
        # Process the first batch then the rest
        for batch in [first]:
            if cache["i"] >= inps.shape[0]:
                break
            try:
                _run_batch(batch)
            except _StopForward:
                break

        for batch in it:
            if cache["i"] >= inps.shape[0]:
                break
            try:
                _run_batch(batch)
            except _StopForward:
                break
    finally:
        # Always remove hook + restore config/state
        hook_handle.remove()
        model.config.use_cache = use_cache
        if model_was_training:
            model.train()

    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    return inps, outs, attention_mask, position_ids

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
    # index of last True (per row)
    idx = sort_mask.sum(dim=1, keepdims=True) - 1
    idx = torch.clamp(idx, min=0)
    thres = torch.gather(sort_res[0], dim=1, index=idx)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask == True).sum().float() / W_mask.numel()
    return W_mask, cur_sparsity
