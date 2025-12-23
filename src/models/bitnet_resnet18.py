import torch
import torch.nn as nn
import torch.nn.functional as F

from src.quant.bitlinear import BitLinear
from src.ops.base2_softmax import Base2Softmax


def _basic_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
    layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
    layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class BitNetResNet18(nn.Module):
    def __init__(self, num_classes: int = 10, softmax_type: str = "standard", temperature: float = 1.0):
        super().__init__()
        self.softmax_type = softmax_type

        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, blocks=2)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = BitLinear(512, num_classes)

        if softmax_type == "base2":
            self.softmax = Base2Softmax(temperature=temperature)
        else:
            self.softmax = nn.Softmax(dim=-1)

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = []
        layers.append(_basic_block(in_channels, out_channels, stride=stride))
        for _ in range(1, blocks):
            layers.append(_basic_block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        # Always return logits; external callers can apply softmax if needed
        return logits


