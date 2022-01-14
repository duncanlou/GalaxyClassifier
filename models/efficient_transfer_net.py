from torch import nn
from torchvision import models


# using efficientnet model based transfer learning
class EfficientModel(nn.Module):
    def __init__(self):
        super(EfficientModel, self).__init__()
        self.net = models.efficientnet_b0(pretrained=True)
        self.l1 = nn.Linear(1000, 256)
        self.dropout = nn.Dropout(0.75)
        self.l2 = nn.Linear(256, 3)
        self.relu = nn.LeakyReLU()

    def forward(self, input):
        x = self.silu(self.batchnorm2d(self.conv1(input)))
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.l1(x)))
        x = self.l2(x)
        return x


model = models.efficientnet_b0(pretrained=True)
print(model._conv_stem)
