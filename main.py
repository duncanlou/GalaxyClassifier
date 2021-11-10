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

batch_size = 256
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


def train_loop(dataloader, model, loss_fn, optimizer):
    print("进入train_loop了！")
    model.train()
    running_loss = 0.0
    size = len(dataloader.sampler)
    for minibatch, (images, labels) in enumerate(dataloader):
        # Compute prediction and loss:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = loss_fn(output, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item
        if minibatch % 10 == 9:  # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, minibatch + 1, running_loss / 10))

    print('Finished Training')


def valid_loop(dataloader, model, loss_fn):
    # valid_loss, val_correct = 0.0, 0
    model.eval()
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


for fold, (train_idx, val_idx) in enumerate(
        skf.split(full_dataset.fits_folder.samples, full_dataset.fits_folder.targets)):
    print('Fold {}'.format(fold + 1))

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=test_sampler)

    for t in range(2):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer)
        valid_loop(test_loader, model, loss_fn)
    print(f"Fold {fold + 1} is Done!")

