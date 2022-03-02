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


# class LouNet(nn.Module):
#     def __init__(self):
#         super(LouNet, self).__init__()
#         self.b1 = nn.Sequential(nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(),
#                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
#
#         self.b2 = nn.Sequential(*self.resnet_block(64, 64, 2, first_block=True))
#         self.b3 = nn.Sequential(*self.resnet_block(64, 128, 2))
#         self.b4 = nn.Sequential(*self.resnet_block(128, 256, 2))
#         self.b5 = nn.Sequential(*self.resnet_block(256, 512, 2))
#
#         self.fc = nn.Linear(512, 3)
#
#     def resnet_block(self, input_channels, num_channels, num_residuals, first_block=False):
#         blk = []
#         for i in range(num_residuals):
#             if i == 0 and not first_block:
#                 blk.append(ResBlock(input_channels, num_channels, use_1x1conv=True, strides=2))
#             else:
#                 blk.append(ResBlock(num_channels, num_channels))
#         return blk
#
#     def forward(self, X):
#         X = self.b1(X)
#         X = self.b2(X)
#         X = self.b3(X)
#         X = self.b4(X)
#         X = self.b5(X)
#         X = F.adaptive_avg_pool2d(X, (1, 1))
#         X = torch.flatten(X)
#         X = self.fc(X)
#         return X

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(ResBlock(input_channels, num_channels, use_1x1conv=True, strides=2))
    else:
        blk.append(ResBlock(num_channels, num_channels))
    return blk


b1 = nn.Sequential(nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, 10))

X = torch.rand(size=(1, 5, 240, 240))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
