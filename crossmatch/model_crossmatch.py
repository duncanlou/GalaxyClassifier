import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RadioOpticalCrossmatchModel(nn.Module):
    def __init__(self):
        super(RadioOpticalCrossmatchModel, self).__init__()
        self.cnn = models.resnet18(pretrained=True).to(device)
        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 20)

        self.cnn_opt = models.resnet18(pretrained=True).to(device)
        self.cnn_opt.conv1 = nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.cnn_opt.fc = nn.Linear(self.cnn_opt.fc.in_features, 10)

        self.fc1 = nn.Linear(20 + 10 + 3 + 3, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, radio_img_data, ps_img_data, ps_source_class_probs, ps_cutouts_pos_info):
        x1 = self.cnn(radio_img_data)
        x2 = self.cnn_opt(ps_img_data)

        x = torch.cat((x1, x2, ps_source_class_probs, ps_cutouts_pos_info), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
