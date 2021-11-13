import torch
import torch.optim as optim
import torchvision.utils
from astropy.table import Table
from torch import nn
import torch.utils.data as data
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from CelebDataset import CelebDataset
from models.GalaxyNet import GalaxyNet
import d2l
from utils import matplotlib_imshow

writer = SummaryWriter()
T = Table.read("data/DuncanSDSSdata.tbl", format="ascii.ipac")  # cost about 35 seconds

tfs = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])

batch_size = 128
number_labels = 3
num_epochs = 10

print(torch.__version__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

IMG_ROOT = "images14"
# 加载图像数据集，并在加载的时候对图像施加变换
full_dataset = CelebDataset(IMG_ROOT, source_table=T, transform=tfs)

# Instantiate a neural network model
model = GalaxyNet().to(device)

train_set_size = int(len(full_dataset) * 0.8)
test_set_size = len(full_dataset) - train_set_size
train_set, test_set = data.random_split(full_dataset, [train_set_size, test_set_size])

trainloader = data.DataLoader(train_set, batch_size, shuffle=True, num_workers=2)
testloader = data.DataLoader(test_set, batch_size, shuffle=True, num_workers=2)

print("Full set size:", len(full_dataset))
print("Train set size: ", train_set_size)
print("Test set size: ", test_set_size)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

dataiter = iter(trainloader)
images, labels = dataiter.__next__()


def evaluate_accuracy_gpu(net, dataloader, device=None):
    """Compute the accuracy for a model on a dataset using a GPU."""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)

    with torch.no_grad():
        for X, y in dataloader:
            if isinstance(X, list):  # Required for BERT Fine-tuning
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]



def train_loop(trainloader, net, loss_fn, optimizer):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform(m.weight)

    net.apply(init_weights)
    print('training on', device)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train_acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(trainloader)

    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(trainloader):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss_fn(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
                timer.stop()
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                    animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, testloader)
        animator.add(epoch + 1, (None, None, test_acc))

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, ' f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec ' f'on {str(device)}')

    print('Finished Training')


def test_loop(dataloader, net, loss_fn):
    net.eval()
    size = len(dataloader.sampler)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = net(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for epoch in range(2):
    running_loss = 0.0
    train_loop(trainloader, model, loss_fn, optimizer, epoch)
    test_loop(testloader, model, loss_fn)
