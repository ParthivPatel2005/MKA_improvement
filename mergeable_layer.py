"""Mergeable layer utilities for learnable-alpha MKA experiments.

These modules wrap two frozen transformer blocks and expose a trainable mixing
coefficient (``alpha``) that controls how their outputs are blended during the
forward pass. An optional MLP-based variant can infer the coefficient from
activation statistics.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class MergeableLayer(nn.Module):
    """Blend two transformer layers with a learnable scalar weight."""

    def __init__(
        self,
        layer_l: nn.Module,
        layer_m: nn.Module,
        alpha_init: float = 0.5,
        use_logit_param: bool = True,
    ) -> None:
        super().__init__()

        self.layer_l = layer_l
        self.layer_m = layer_m
        self.use_logit_param = use_logit_param

        for param in self.layer_l.parameters():
            param.requires_grad = False
        for param in self.layer_m.parameters():
            param.requires_grad = False

        alpha_init = float(alpha_init)
        alpha_init = min(max(alpha_init, 1e-4), 1.0 - 1e-4)

        if use_logit_param:
            logit_init = torch.logit(torch.tensor(alpha_init, dtype=torch.float32))
            self.alpha_logit = nn.Parameter(logit_init)
        else:
            self.alpha_raw = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    @property
    def alpha(self) -> torch.Tensor:
        if self.use_logit_param:
            return torch.sigmoid(self.alpha_logit)
        return torch.clamp(self.alpha_raw, 0.0, 1.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        alpha = self.alpha

        output_l = self.layer_l(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        output_m = self.layer_m(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        if isinstance(output_l, tuple):
            hidden_l = output_l[0]
            hidden_m = output_m[0]
            blended_hidden = alpha * hidden_l + (1.0 - alpha) * hidden_m
            return (blended_hidden,) + output_l[1:]

        return alpha * output_l + (1.0 - alpha) * output_m

    def extra_repr(self) -> str:
        value = self.alpha.detach().float().cpu().item()
        return f"alpha={value:.4f}"


class MLPMergeableLayer(nn.Module):
    """Variant that predicts ``alpha`` via a small MLP."""

    def __init__(
        self,
        layer_l: nn.Module,
        layer_m: nn.Module,
        hidden_dim: int = 64,
        input_features: int = 4,
    ) -> None:
        super().__init__()

        self.layer_l = layer_l
        self.layer_m = layer_m

        for param in self.layer_l.parameters():
            param.requires_grad = False
        for param in self.layer_m.parameters():
            param.requires_grad = False

        self.alpha_mlp = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _compute_stats(hidden_states: torch.Tensor) -> torch.Tensor:
        flat = hidden_states.view(-1, hidden_states.size(-1))
        mean = flat.mean(dim=0).mean()
        std = flat.std(dim=0).mean()
        min_val = flat.min()
        max_val = flat.max()
        return torch.stack([mean, std, min_val, max_val])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        features = self._compute_stats(hidden_states).to(hidden_states.device)
        alpha = self.alpha_mlp(features)

        output_l = self.layer_l(hidden_states, attention_mask=attention_mask, **kwargs)
        output_m = self.layer_m(hidden_states, attention_mask=attention_mask, **kwargs)

        if isinstance(output_l, tuple):
            hidden_l = output_l[0]
            hidden_m = output_m[0]
            blended_hidden = alpha * hidden_l + (1.0 - alpha) * hidden_m
            return (blended_hidden,) + output_l[1:]

        return alpha * output_l + (1.0 - alpha) * output_m


def create_mergeable_layer(
    layer_l: nn.Module,
    layer_m: nn.Module,
    alpha_init: float = 0.5,
    mode: str = "simple",
) -> nn.Module:
    if mode == "simple":
        return MergeableLayer(layer_l, layer_m, alpha_init=alpha_init)
    if mode == "mlp":
        return MLPMergeableLayer(layer_l, layer_m)
    raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    block_a = nn.Linear(16, 16)
    block_b = nn.Linear(16, 16)
    wrapper = MergeableLayer(block_a, block_b, alpha_init=0.6)
    x = torch.randn(2, 5, 16)
    out = wrapper(x)
    print("Output shape:", out.shape)
    print("Trainable params:", sum(p.numel() for p in wrapper.parameters() if p.requires_grad))
