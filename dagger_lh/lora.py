"""Minimal LoRA implementation with support for nn.Linear and nn.LSTM.

Why hand-rolled rather than `peft`: the policies here are robomimic networks whose
recurrent core keeps its weights as raw `nn.Parameter`s on `nn.LSTM` (not `nn.Linear`
submodules), so an off-the-shelf Linear-only injector cannot adapt the part of a
BC-RNN that matters most. `LoRALSTM` reparameterises `weight_ih_l*` / `weight_hh_l*`
with low-rank deltas while still dispatching to the fused cuDNN/MPS LSTM kernel.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wraps a frozen `nn.Linear` with a trainable rank-`r` residual branch."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.scaling = float(alpha) / max(1, int(rank))
        self.lora_A = nn.Parameter(torch.zeros(self.rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # A ~ Kaiming, B = 0  =>  the adapter starts as an exact no-op.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self._merged = False

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self._merged:
            return out
        delta = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + self.scaling * delta

    @torch.no_grad()
    def merge(self) -> None:
        """Fold the adapter into the base weight (for fast inference / export)."""
        if self._merged:
            return
        self.base.weight.add_(self.scaling * (self.lora_B @ self.lora_A))
        self._merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self._merged:
            return
        self.base.weight.sub_(self.scaling * (self.lora_B @ self.lora_A))
        self._merged = False

    def extra_repr(self) -> str:
        return f"rank={self.rank}, scaling={self.scaling:.3f}"


class LoRALSTM(nn.LSTM):
    """`nn.LSTM` whose weight matrices carry additive low-rank deltas.

    `nn.LSTM` reads its weights out of the cached `_flat_weights` list, so simply
    registering a parametrisation on `weight_ih_l0` would be silently ignored.
    Instead we rebuild `_flat_weights` from the adapted tensors on every forward,
    which keeps autograd intact and still hits the fused kernel.
    """

    def __init__(self, base: nn.LSTM, rank: int = 8, alpha: float = 16.0):
        # Re-create the same module shape, then adopt the base module's parameters.
        super().__init__(
            input_size=base.input_size,
            hidden_size=base.hidden_size,
            num_layers=base.num_layers,
            bias=base.bias,
            batch_first=base.batch_first,
            dropout=base.dropout,
            bidirectional=base.bidirectional,
        )
        with torch.no_grad():
            for name, p in base.named_parameters():
                getattr(self, name).copy_(p)
        for p in self.parameters():
            p.requires_grad_(False)

        self.rank = int(rank)
        self.scaling = float(alpha) / max(1, int(rank))
        self._adapted: List[str] = []
        self.lora_params = nn.ParameterDict()
        for name, p in list(self.named_parameters()):
            if not name.startswith("weight_"):
                continue
            out_f, in_f = p.shape
            key = name.replace(".", "__")
            a = nn.Parameter(torch.zeros(self.rank, in_f))
            b = nn.Parameter(torch.zeros(out_f, self.rank))
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            self.lora_params[key + "_A"] = a
            self.lora_params[key + "_B"] = b
            self._adapted.append(name)

    def _adapted_weight(self, name: str) -> torch.Tensor:
        base = getattr(self, name)
        if name not in self._adapted:
            return base
        key = name.replace(".", "__")
        a = self.lora_params[key + "_A"]
        b = self.lora_params[key + "_B"]
        return base + self.scaling * (b @ a)

    def _refresh_flat_weights(self) -> None:
        self._flat_weights = [
            self._adapted_weight(n) if isinstance(n, str) and hasattr(self, n) else None
            for n in self._flat_weights_names
        ]

    def forward(self, *args, **kwargs):  # type: ignore[override]
        self._refresh_flat_weights()
        return super().forward(*args, **kwargs)

    def flatten_parameters(self) -> None:  # noqa: D102
        # Disabled: the adapted weights are non-leaf tensors and cannot live in a
        # flat cuDNN buffer. Harmless on CPU/MPS; avoids a warning on CUDA.
        return


# --------------------------------------------------------------------------------------
# Injection / extraction helpers
# --------------------------------------------------------------------------------------

def _iter_named_modules(root: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    return list(root.named_modules())


def _matches(name: str, substrings: List[str]) -> bool:
    return (not substrings) or any(s in name for s in substrings)


def inject_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_substrings: Optional[List[str]] = None,
    adapt_rnn: bool = True,
    min_dim: int = 8,
) -> Dict[str, int]:
    """Replace eligible submodules of `model` in-place with LoRA-wrapped versions.

    Returns a summary count of what was adapted. Idempotent: already-wrapped
    modules are skipped.
    """
    subs = list(target_substrings or [])
    stats = {"linear": 0, "lstm": 0, "small_full_ft": 0, "skipped_mha": 0}

    for name, module in _iter_named_modules(model):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, (LoRALinear, LoRALSTM)):
                continue
            if not _matches(full, subs):
                continue
            # `nn.MultiheadAttention` reads `out_proj.weight` directly in its fused
            # forward path, so a wrapped module would raise. Its q/k/v also live in a
            # packed `in_proj_weight` Parameter that a Linear injector cannot reach --
            # use `dagger_lh.policies.act` transformer blocks for full coverage.
            if isinstance(module, nn.MultiheadAttention):
                stats["skipped_mha"] += 1
                continue
            if isinstance(child, nn.Linear):
                if min(child.in_features, child.out_features) < min_dim:
                    # A rank-r adapter on e.g. a 400->5 head costs as much as the
                    # weight itself, so flag it for plain full finetuning instead.
                    child._lora_full_finetune = True  # noqa: SLF001
                    stats["small_full_ft"] += 1
                    continue
                r = min(rank, min(child.in_features, child.out_features))
                setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
                stats["linear"] += 1
            elif adapt_rnn and isinstance(child, nn.LSTM):
                setattr(module, child_name, LoRALSTM(child, rank, alpha))
                stats["lstm"] += 1
    return stats


def mark_trainable(
    model: nn.Module,
    train_biases: bool = True,
    train_output_head: bool = False,
    head_substrings: Tuple[str, ...] = ("output", "action_head", "decoder.out"),
) -> None:
    """Freeze everything except LoRA parameters (plus optional biases / head)."""
    for p in model.parameters():
        p.requires_grad_(False)
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name or "lora_params" in name:
            p.requires_grad_(True)
        elif train_biases and name.endswith("bias"):
            p.requires_grad_(True)
        elif train_output_head and any(h in name for h in head_substrings):
            p.requires_grad_(True)
    # Layers too small to be worth adapting are trained directly.
    for m in model.modules():
        if getattr(m, "_lora_full_finetune", False):
            for p in m.parameters(recurse=False):
                p.requires_grad_(True)


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Every *trainable* tensor: adapters, unfrozen biases and small full-FT heads.

    This is the delta that gets checkpointed per DAgger round — a few hundred KB,
    so keeping one snapshot per round for the whole session is cheap.
    """
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    sd = model.state_dict()
    return {k: sd[k].detach().cpu().clone() for k in trainable if k in sd}


def load_lora_state_dict(model: nn.Module, sd: Dict[str, torch.Tensor],
                         strict: bool = False) -> None:
    missing = model.load_state_dict(sd, strict=False)
    if strict and missing.unexpected_keys:
        raise RuntimeError(f"unexpected LoRA keys: {missing.unexpected_keys}")


def reset_lora(model: nn.Module) -> None:
    """Re-initialise every adapter back to a no-op (B=0)."""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                nn.init.kaiming_uniform_(m.lora_A, a=math.sqrt(5))
                nn.init.zeros_(m.lora_B)
            elif isinstance(m, LoRALSTM):
                for k, p in m.lora_params.items():
                    if k.endswith("_A"):
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    else:
                        nn.init.zeros_(p)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def describe(model: nn.Module) -> str:
    tr, tot = count_parameters(model)
    pct = 100.0 * tr / max(1, tot)
    n_lin = sum(isinstance(m, LoRALinear) for m in model.modules())
    n_rnn = sum(isinstance(m, LoRALSTM) for m in model.modules())
    n_full = sum(getattr(m, "_lora_full_finetune", False) for m in model.modules())
    return (f"LoRA: {n_lin} linear + {n_rnn} lstm adapted, {n_full} small full-FT | "
            f"trainable {tr:,}/{tot:,} ({pct:.2f}%)")
