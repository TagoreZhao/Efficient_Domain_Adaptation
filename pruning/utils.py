import math
import torch
import torch.nn as nn


def get_comp_norm(comp, dim):
    comp_weight = comp.base_layer.weight + torch.matmul(comp.lora_B.default.weight,
                                                        comp.lora_A.default.weight)
    return torch.norm(comp_weight, p=2, dim=dim).reshape(1, -1)

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
    """
    Return the decoder layer ModuleList for Qwen3-style models, including PEFT-wrapped models.

    Works for:
      - base Qwen3ForCausalLM:            model.model.layers
      - PEFT wrappers (common):          model.base_model.model.model.layers
                                        model.base_model.model.layers
                                        model.base_model.model.layers, etc.
    """
    # 1) Unwrap PEFT if present
    base = model
    if hasattr(base, "base_model"):
        base = base.base_model
        # PEFT often stores the underlying HF model under .model
        if hasattr(base, "model"):
            base = base.model

    # 2) Try common Qwen-style paths
    if hasattr(base, "model") and hasattr(base.model, "layers"):
        return base.model.layers

    # Some stacks use model.layers directly
    if hasattr(base, "layers"):
        return base.layers

    # 3) Last resort: search for a ModuleList named "...layers"
    for name, mod in base.named_modules():
        if name.endswith("layers") and isinstance(mod, nn.ModuleList):
            return mod

    raise AttributeError(
        "Could not find decoder layers. Tried Qwen-style paths and fallback search "
        "for a ModuleList named '*layers'."
    )

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
    Qwen3 expects `position_embeddings` as a tuple (cos, sin) from rotary_emb.

    Works for:
      - base models: model.model.rotary_emb
      - PEFT models: model.base_model.model.model.rotary_emb (or similar)
    """
    base = _unwrap_to_base_model(model)

    # Common Qwen-style locations
    if hasattr(base, "model") and hasattr(base.model, "rotary_emb"):
        rotary = base.model.rotary_emb
    elif hasattr(base, "rotary_emb"):
        rotary = base.rotary_emb
    else:
        raise AttributeError(
            "Could not find rotary_emb. Tried base.model.rotary_emb and base.rotary_emb "
            "(with PEFT unwrapping)."
        )

    # Different HF implementations vary slightly in signature; support both common call styles.
    try:
        return rotary(hidden_states, position_ids)  # (cos, sin)
    except TypeError:
        # Some variants use keyword args
        return rotary(x=hidden_states, position_ids=position_ids)


def layer_forward(model, layer, hidden_states, attention_mask=None, position_ids=None):
    """
    Forward a single Qwen3 decoder layer correctly by providing position_embeddings.
    Compatible with PEFT-wrapped models; `layer` should be the actual decoder block module.

    hidden_states: [bsz, seqlen, hidden]
    Returns: hidden_states_out [bsz, seqlen, hidden]
    """
    bsz, seqlen, _ = hidden_states.shape
    dev = hidden_states.device

    pos_ids = make_position_ids(position_ids, seqlen, bsz, dev)
    pos_emb = compute_position_embeddings(model, hidden_states, pos_ids)

    out = layer(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=pos_ids,
        position_embeddings=pos_emb,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
    )
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
class _StopForward(Exception):
    """Internal exception used to early-exit the forward pass after capturing activations."""
    pass


def _unwrap_to_base_model(model: nn.Module) -> nn.Module:
    """
    If `model` is a PEFT wrapper, unwrap to the underlying Hugging Face model.
    This does not detach adapters; it just returns the object that owns the transformer stack.
    """
    # Common PEFT wrappers expose .base_model (PeftModelForCausalLM, etc.)
    if hasattr(model, "base_model"):
        base = model.base_model
        # In PEFT, base_model often has .model which is the underlying HF model
        if hasattr(base, "model"):
            return base.model
        return base
    return model


def _get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """
    Find the decoder layer ModuleList for Qwen-style / Llama-style HF models,
    handling both plain models and PEFT wrappers.
    """
    base = _unwrap_to_base_model(model)

    # Try common attribute chains
    candidates = [
        ("model", "layers"),              # e.g., Qwen3ForCausalLM: base.model.layers
        ("model", "model", "layers"),      # some models nest .model.model.layers
        ("transformer", "h"),              # GPT-style: transformer.h
        ("gpt_neox", "layers"),            # GPT-NeoX style
        ("decoder", "layers"),             # encoder-decoder style (decoder.layers)
    ]

    for chain in candidates:
        obj = base
        ok = True
        for attr in chain:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and isinstance(obj, nn.ModuleList):
            return obj

    # Last resort: search by name for something ending with ".layers"
    for name, mod in base.named_modules():
        if name.endswith("layers") and isinstance(mod, nn.ModuleList):
            return mod

    raise AttributeError(
        "Could not find decoder layers ModuleList. "
        "Tried common paths (model.layers, model.model.layers, transformer.h, etc.)."
    )


@torch.no_grad()
def prepare_calibration_input(model, dataloader, device=None, max_samples=None):
    """
    Capture hidden-states entering the first decoder layer for calibration samples.
    Works with PEFT-wrapped models by attaching the hook to the underlying base layer.
    (This is edited based on Wanda Implementation)
    Returns:
        inps:           [n_to_capture, seqlen, hidden]
        outs:           same shape as inps (zeros)
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

    def _get_input_ids(batch):
        # Supports:
        # - (input_ids, labels) tuples
        # - dict batches with "input_ids"
        # - direct tensors
        if isinstance(batch, dict):
            x = batch["input_ids"]
        elif isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    def _get_attention_mask(batch):
        if isinstance(batch, dict) and "attention_mask" in batch:
            return batch["attention_mask"]
        return None

    # Peek one batch for seqlen
    it = iter(dataloader)
    first = next(it)
    first_ids = _get_input_ids(first)
    seqlen = first_ids.shape[1]

    # Determine number of samples to capture
    total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
    n_to_capture = total_batches if total_batches is not None else 0
    if max_samples is not None:
        n_to_capture = min(n_to_capture, max_samples) if n_to_capture else max_samples
    if n_to_capture == 0:
        if max_samples is None:
            raise ValueError("Dataloader has no __len__; pass max_samples.")
        n_to_capture = max_samples

    dtype = next(model.parameters()).dtype
    hidden = model.config.hidden_size

    inps = torch.zeros((n_to_capture, seqlen, hidden), dtype=dtype, device=device)
    outs = torch.zeros_like(inps)

    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    def _pre_hook(module, args, kwargs):
        if not args:
            raise RuntimeError("Decoder layer received no hidden_states positional arg.")

        hs = args[0]
        # hs: [bsz, seqlen, hidden]
        if hs.device != device:
            hs = hs.to(device, non_blocking=True)

        bsz = hs.shape[0]
        for b in range(bsz):
            if cache["i"] >= inps.shape[0]:
                break
            inps[cache["i"]] = hs[b]
            cache["i"] += 1

        cache["attention_mask"] = kwargs.get("attention_mask", cache["attention_mask"])
        cache["position_ids"]   = kwargs.get("position_ids", cache["position_ids"])

        if cache["i"] >= inps.shape[0]:
            raise _StopForward()

    # Attach hook to the *actual* first decoder layer module
    hook_handle = layers[0].register_forward_pre_hook(_pre_hook, with_kwargs=True)

    def _run_batch(batch):
        input_ids = _get_input_ids(batch).to(device, non_blocking=True)
        attn_mask = _get_attention_mask(batch)
        if attn_mask is not None:
            attn_mask = attn_mask.to(device, non_blocking=True)

        # Run through the (possibly PEFT-wrapped) model.
        # The hook is on base layer, so it will fire.
        model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            use_cache=False,
        )

    try:
        # First batch then remainder
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
        hook_handle.remove()
        model.config.use_cache = use_cache
        if model_was_training:
            model.train()

    return inps, outs, cache["attention_mask"], cache["position_ids"]

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

@torch.no_grad()
def merge_lora_into_base(lora_linear, adapter: str = "default"):
    """
    base_weight <- base_weight + delta(adapter)
    Does NOT change requires_grad settings.
    """
    if adapter not in lora_linear.lora_A:
        return

    # PEFT LoRA Linear exposes get_delta_weight(adapter) in recent versions;
    # if not, you can compute B @ A * scaling yourself.
    if hasattr(lora_linear, "get_delta_weight"):
        delta = lora_linear.get_delta_weight(adapter)
    else:
        print("Computing delta manually")
        A = lora_linear.lora_A[adapter].weight
        B = lora_linear.lora_B[adapter].weight
        delta = (B @ A) * lora_linear.scaling[adapter]
        # fan_in_fan_out handling if needed
        if getattr(lora_linear, "fan_in_fan_out", False):
            delta = delta.T

    # Make sure delta dtype/device matches base weight
    base_w = lora_linear.base_layer.weight
    base_w.add_(delta.to(device=base_w.device, dtype=base_w.dtype))

@torch.no_grad()
def reset_lora(lora_linear, adapter: str = "default"):
    """
    Re-initialize LoRA weights in-place.
    Typical LoRA init: A ~ Kaiming, B = 0.
    """
    if adapter not in lora_linear.lora_A:
        return
    A = lora_linear.lora_A[adapter].weight
    B = lora_linear.lora_B[adapter].weight

    nn.init.kaiming_uniform_(A, a=math.sqrt(5))
    nn.init.zeros_(B)