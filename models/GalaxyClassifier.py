import torch
import torch.nn as nn
import torch.nn.functional as F


# Define a convolution neural network
class GalaxyClassifier(nn.Module):
    def __init__(self):
        super(GalaxyClassifier, self).__init__()

        self.conv1 = nn.Conv2d(5, 20, (5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 64, (5, 5))

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(64 * 57 * 57, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.pool(F.relu(self.conv1(x)))
        # If the size is a square, you can specify with a single number
        x = self.pool(F.relu(self.conv2(x)))
        # flatten all dimensions except the batch dimension
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu((self.fc2(x)))
        x = self.fc3(x)

        return x


