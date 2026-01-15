import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Union

from peft.tuners.lora import Linear

def transpose(weight: torch.Tensor, fan_in_fan_out: bool) -> torch.Tensor:
    return weight.T if fan_in_fan_out else weight

class MaskedLoraLinear(Linear):
    """
    Unstructured masked LoRA:
      W_eff = (W_base + sum(delta_W_adapter)) ⊙ mask
    Note: This materializes dense delta weights (unavoidable for unstructured masking).
    """

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        mask: Optional[torch.Tensor] = None,
        score: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            base_layer=base_layer,
            adapter_name=adapter_name,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=fan_in_fan_out,
            is_target_conv_1d_layer=is_target_conv_1d_layer,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
            **kwargs,
        )

        w = self.base_layer.weight

        # Buffers: saved in state_dict and moved/cast with the module automatically.
        if mask is None:
            mask = torch.ones_like(w)
        else:
            if mask.shape != w.shape:
                raise ValueError(f"mask shape {mask.shape} must match weight shape {w.shape}")
            mask = mask.to(dtype=w.dtype)

        if score is None:
            score = torch.zeros_like(w)
        else:
            if score.shape != w.shape:
                raise ValueError(f"score shape {score.shape} must match weight shape {w.shape}")
            score = score.to(dtype=w.dtype)

        self.register_buffer("mask", mask, persistent=True)
        self.register_buffer("score", score, persistent=True)

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        A = self.lora_A[adapter].weight
        B = self.lora_B[adapter].weight
        delta = (B @ A) * self.scaling[adapter]
        return transpose(delta, self.fan_in_fan_out)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)

        if adapter_names is not None:
            return self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)

        if self.merged:
            return self.base_layer(x, *args, **kwargs)

        w_eff = self.base_layer.weight
        for adapter in self.active_adapters:
            if adapter in self.lora_A:
                w_eff = w_eff + self.get_delta_weight(adapter)

        w_eff = w_eff * self.mask
        bias = getattr(self.base_layer, "bias", None)
        return F.linear(x, w_eff, bias)

