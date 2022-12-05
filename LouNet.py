import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    """The Residual block of ResNet."""

    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class LouNet(nn.Module):
    def __init__(self):
        super(LouNet, self).__init__()
        self.b1 = nn.Sequential(nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b2 = nn.Sequential(*self.resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*self.resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*self.resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*self.resnet_block(256, 512, 2))

        self.fc1 = nn.Linear(512 + 3, 48)
        self.fc2 = nn.Linear(48, 3)

    def resnet_block(self, input_channels, num_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(ResBlock(input_channels, num_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(ResBlock(num_channels, num_channels))
        return blk

    def forward(self, input):
        output = self.b1(input)
        output = self.b2(output)
        output = self.b3(output)
        output = self.b4(output)
        output = self.b5(output)
        output = F.adaptive_avg_pool2d(output, (1, 1))
        output = torch.flatten(output)
        output = self.fc1(output)
        output = F.relu(output)
        output = self.fc2(output)
        return output
