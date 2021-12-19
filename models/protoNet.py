from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary


# Define a convolution neural network
class GalaxyNet(nn.Module):
    def __init__(self, l1=120, l2=84):
        super(GalaxyNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(60 * 60 * 32, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# net = GalaxyNet()
# torchsummary.summary(net, input_size=(5, 240, 240), device="cpu")
