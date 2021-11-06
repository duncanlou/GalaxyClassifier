import time

import torch
import torch.optim as optim
from astropy.table import Table
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms


from CelebDataset import CelebDataset
from models.GalaxyNet import GalaxyNet

T = Table.read("data/DuncanSDSSdata.tbl", format="ascii.ipac")  # cost about 35 seconds

tfs = transforms.Compose([
    transforms.ToTensor()
])

batch_size = 512
number_of_labels = 3

print(torch.__version__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
torch.manual_seed(42)

# 加载图像数据集，并在加载的时候对图像施加变换
full_dataset = CelebDataset("data/images14", source_table=T, transform=tfs)

# Instantiate a neural network model
model = GalaxyNet().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
num_epochs = 2
skf = StratifiedKFold(shuffle=True, random_state=42)  # divides dataset into 5 folds by default
foldperf = {}


def train_loop(dataloader, model, loss_fn, optimizer):
    print("进入train_loop了！")
    # model.train()
    # train_loss, train_correct = 0.0, 0.0

    start = time.time()
    size = len(dataloader.sampler)
    for minibatch, (images, labels) in enumerate(dataloader):
        print(f"在train_loop中的for循环当中，minibatch = {minibatch}")
        # Compute prediction and loss:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        # hasNullValue = np.isnan(images).any()
        loss = loss_fn(output, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), minibatch * len(images)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        # train_loss += loss.item() * images.size(0)  # images.size(): torch.Size([30, 5, 240, 240])
        # scores, predictions = torch.max(output.data, 1)
        # train_correct += (predictions == labels).sum().item()

    end = time.time()
    print("train_loop中for循环的时间： ", end - start)

    # return train_loss, train_correct



def valid_loop(dataloader, model, loss_fn):
    # valid_loss, val_correct = 0.0, 0
    # model.eval()
    size = len(dataloader.sampler)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    # for images, labels in dataloader:
    #     images, labels = images.to(device), labels.to(device)
    #     output = model(images)
    #     loss = loss_fn(output, labels)
    #     valid_loss += loss.item() * images.size(0)
    #     scores, predictions = torch.max(output.data, 1)
    #     val_correct += (predictions == labels).sum().item()

    # return valid_loss, val_correct


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


    # history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

    # for epoch in range(num_epochs):
    #     print(f'epoch: {epoch}')
    #     train_loss, train_correct = train_loop(train_loader, model, loss_fn, optimizer)
    #     test_loss, test_correct = valid_loop(test_loader, model, loss_fn)
    #
    #     train_loss = train_loss / len(train_loader.sampler)
    #     train_acc = train_correct / len(train_loader.sampler) * 100
    #     test_loss = test_loss / len(test_loader.sampler)
    #     test_acc = test_correct / len(test_loader.sampler) * 100
    #
    #     print(
    #         "Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(
    #             epoch + 1,
    #             num_epochs,
    #             train_loss,
    #             test_loss,
    #             train_acc,
    #             test_acc)
    #     )
    #     history['train_loss'].append(train_loss)
    #     history['test_loss'].append(test_loss)
    #     history['train_acc'].append(train_acc)
    #     history['test_acc'].append(test_acc)

    # foldperf['fold{}'.format(fold + 1)] = history

    for t in range(num_epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer)
        valid_loop(test_loader, model, loss_fn)
    print(f"Fold {fold + 1} is Done!")

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

