from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary


# Define a convolution neural network
class GalaxyNet(nn.Module):
    def __init__(self, l1=120, l2=84):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=(5, 5)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64 * 6 * 6, l1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(l1, l2),
            nn.ReLU(inplace=True),
            nn.Linear(l2, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# net = GalaxyNet()
# torchsummary.summary(net, input_size=(5, 240, 240), device="cpu")

