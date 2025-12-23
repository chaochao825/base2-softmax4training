import math
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F


class Base2Softmax(nn.Module):
    """Softmax that directly uses base-2 exponent: 2^x with stability guard."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / self.temperature
        x = x - x.amax(dim=-1, keepdim=True)
        exp_x = torch.exp2(x)
        return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)


def base2_softmax_fn(x: torch.Tensor, dim: int | None = None, temperature: float = 1.0):
    """Functional base-2 softmax (2^x) with numerical stability."""
    if dim is None:
        dim = -1
    x = x / float(temperature)
    x = x - x.amax(dim=dim, keepdim=True)
    exp_x = torch.exp2(x)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


@contextmanager
def patch_torch_softmax_base2(temperature: float = 1.0):
    """
    Temporarily patch torch.nn.functional.softmax and cross_entropy to use base-2.

    Useful for HF attention modules and standard training loops.
    """
    orig_softmax = F.softmax
    orig_cross_entropy = F.cross_entropy
    LOG_2 = math.log(2)

    def _base2_softmax(x, dim=None, _stacklevel=3, dtype=None):
        return base2_softmax_fn(x, dim=dim, temperature=temperature)

    def _base2_cross_entropy(input, target, *args, **kwargs):
        # Scaling logits by ln(2) makes the internal exp(x) behave like 2^x
        return orig_cross_entropy(input * (LOG_2 / temperature), target, *args, **kwargs)

    F.softmax = _base2_softmax  # type: ignore[assignment]
    torch.softmax = _base2_softmax
    F.cross_entropy = _base2_cross_entropy
    
    try:
        yield
    finally:
        F.softmax = orig_softmax  # type: ignore[assignment]
        torch.softmax = orig_softmax
        F.cross_entropy = orig_cross_entropy
