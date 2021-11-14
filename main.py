import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
writer = SummaryWriter('runs/MyResearchProject_Duncan')

import torchvision.utils
from torchvision import transforms

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from astropy.table import Table

# import from local project
from utils import matplotlib_show_source_img, matplotlib_imshow
from CelebDataset import CelebDataset
from models.GalaxyNet import GalaxyNet
import d2l


print(torch.__version__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

T = Table.read("data/DuncanSDSSdata.tbl", format="ascii.ipac")  # cost about 35 seconds
IMG_ROOT = "data/images14"

batch_size = 256
num_class = 3
num_epochs = 10


tfs = transforms.Compose([
    transforms.ToTensor(),
])

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

dataiter = iter(train_set)
images, labels = dataiter.__next__()
img_list = []
for i in range(5):
    img_list.append(images[i])
img_grid = torchvision.utils.make_grid(img_list)
matplotlib_imshow(img_grid, one_channel=True)
matplotlib_show_source_img(images)



# writer.add_graph(model)
# writer.close()


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


train_loop(trainloader, model, loss_fn, optimizer)
