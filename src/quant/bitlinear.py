import torch
import torch.nn as nn
import torch.nn.functional as F


class BitLinear(nn.Module):
    """BitNet 1.58-bit Linear Layer with ternary weights {-1, 0, 1}.

    Training uses a straight-through estimator. Inference uses hard-quantized weights.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.register_buffer("weight_scale", torch.ones(1))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @staticmethod
    def _quantize_weights(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Scale based on mean absolute value with numerical floor
        scale = weights.abs().mean().clamp(min=1e-5)
        # Ternarize to {-1, 0, 1}. Threshold chosen as scale / 3 per common BitNet practice
        mask = (weights.abs() > scale / 3).to(weights.dtype)
        weights_q = torch.sign(weights) * mask
        return weights_q, scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q, scale = self._quantize_weights(self.weight)
        if self.training:
            # Straight-through estimator
            w_q = w_q + self.weight - self.weight.detach()
        out = F.linear(x, w_q, self.bias)
        return out * scale


def get_bitlinear_quant_stats(module: BitLinear) -> dict:
    with torch.no_grad():
        w_q, scale = module._quantize_weights(module.weight)
        return {
            "weights": module.weight.detach().flatten().cpu().float(),
            "weights_q": w_q.detach().flatten().cpu().float(),
            "scale": float(scale.detach().cpu()),
        }


