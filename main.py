import os

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# import from local project
from FitsImageFolder import FitsImageFolder
from models.GalaxyNet import GalaxyNet
from utils import showImages

print("torch version: ", torch.__version__)
writer = SummaryWriter('runs/source_classifier')

src_root_path = os.path.join(os.getcwd(), "data/sources")

tfs = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop((50, 50))
])

# 加载图像数据集，并在加载的时候对图像施加变换
dataset = FitsImageFolder(root="data/sources", transform=tfs)

train_set_size = int(len(dataset) * 0.8)
validation_set_size = int(len(dataset) * 0.1)
test_set_size = len(dataset) - train_set_size - validation_set_size

train_set, validation_set, test_set = random_split(dataset,
                                                   [train_set_size, validation_set_size, test_set_size])

print("Full set size:", len(dataset))
print("Train set size: ", train_set_size)
print("Validation set size: ", validation_set_size)
print("Test set size: ", test_set_size)


def train_source_classifier(data):
    model = GalaxyNet(120, 84)

    device = 'cpu'
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    print('Using {} device'.format(device))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-5)

    trainset, validset = data

    training_loader = DataLoader(
        trainset,
        # batch_size=int(config["batch_size"]),
        batch_size=64,
        num_workers=2,
        shuffle=True
    )

    validation_loader = DataLoader(
        validset,
        batch_size=64,
        num_workers=2,
        shuffle=False
    )

    num_epochs = 10
    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []

    total_step = len(training_loader)

    for epoch in range(num_epochs):

        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(training_loader):
            if batch_idx == 0:
                showImages(images[0])
            images, labels = images.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if batch_idx % 20 == 19:  # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, batch_idx + 1, running_loss / 20))
                running_loss = 0.0


        # validation loss
        batch_loss = 0
        total_t = 0
        correct_t = 0
        with torch.no_grad():
            model.eval()
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                labels_hat = model(images)
                loss_t = criterion(labels_hat, labels)
                batch_loss += loss_t.item()
                _, predicted = torch.max(labels_hat, dim=1)
                correct_t += torch.sum(predicted == labels).item()
                total_t += labels.size(0)

            val_acc.append(f"{100 * correct_t / total_t} %")
            val_loss.append(batch_loss / len(validation_loader))
            network_learned = batch_loss < valid_loss_min
            print(f"validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f} %\n")
            if network_learned:
                valid_loss_min = batch_loss
                torch.save(model.state_dict(), "model_classification_tutorial.pt")
                print('Detected network improvement, saving current model')
        model.train()

    print('Finished Training')


if __name__ == "__main__":
    train_source_classifier(data=(train_set, validation_set))
