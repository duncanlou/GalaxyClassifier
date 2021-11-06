import numpy as np
import torch
from astropy.table import Table
from torch import nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from torchvision import transforms

from sklearn.model_selection import StratifiedKFold

from CelebDataset import CelebDataset
from models.GalaxyClassifier import GalaxyClassifier

tfs = transforms.Compose([
    transforms.ToTensor()
])

BATCH_SIZE = 512
NUM_LABEL = 3
NUM_EPOCH = 10

print(torch.__version__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


# 加载图像数据集，并在加载的时候对图像施加变换
full_dataset = CelebDataset("data/images14", transform=tfs)

# Instantiate a neural network model
model = GalaxyClassifier().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
skf = StratifiedKFold()  # divides dataset into 10 folds


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    train_loss, train_correct = 0.0, 0.0

    for minibatch, (images, labels) in enumerate(dataloader):
        # Compute prediction and loss:
        images, labels = images.to(device), torch.as_tensor(labels).to(device)
        output = model(images)
        # hasNullValue = np.isnan(images).any()
        loss = loss_fn(output, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)  # images.size(): torch.Size([30, 5, 240, 240])
        scores, predictions = torch.max(output.data, 1)
        train_correct += (predictions == labels).sum().item()

    return train_loss, train_correct


def valid_loop(dataloader, model, loss_fn):
    valid_loss, val_correct = 0.0, 0
    model.eval()

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = loss_fn(output, labels)
        valid_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        val_correct += (predictions == labels).sum().item()

    return valid_loss, val_correct


train_idx, test_idx = skf.split(full_dataset.fits_folder.samples, full_dataset.fits_folder.targets)

for fold, (train_idx, val_idx) in enumerate(
        skf.split(full_dataset.fits_folder.samples, full_dataset.fits_folder.targets)):
    print('Fold {}'.format(fold + 1))

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=test_sampler)

    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

    for epoch in range(num_epochs):
        train_loss, train_correct = train_loop(train_loader, model, loss_fn, optimizer)
        test_loss, test_correct = valid_loop(test_loader, model, loss_fn)

        train_loss = train_loss / len(train_loader.sampler)
        train_acc = train_correct / len(train_loader.sampler) * 100
        test_loss = test_loss / len(test_loader.sampler)
        test_acc = test_correct / len(test_loader.sampler) * 100

        print(
            "Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(
                epoch + 1,
                num_epochs,
                train_loss,
                test_loss,
                train_acc,
                test_acc)
        )
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

    foldperf['fold{}'.format(fold + 1)] = history
