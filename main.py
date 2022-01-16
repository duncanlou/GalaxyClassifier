import copy
import os
import time

import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models

# import from local project
import utils
from FitsImageFolder import FitsImageFolder

print("torch version: ", torch.__version__)

src_root_path = os.path.join(os.getcwd(), "data/sources")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")
torch.cuda.empty_cache()

# Data augmentation and normalization for training
# Just normalization for validation
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=45)])
validation_transforms = transforms.Compose([
    transforms.ToTensor()
])

# 加载图像数据集，并在加载的时候对图像施加变换
dataset_1 = FitsImageFolder(root=src_root_path, transform=train_transforms)
dataset_2 = FitsImageFolder(root=src_root_path, transform=validation_transforms)
dataset_3 = FitsImageFolder(root=src_root_path, transform=validation_transforms)

dataset_indices = list(range(len(dataset_1)))
np.random.shuffle(dataset_indices)

test_split_index = int(np.floor(0.2 * len(dataset_1)))
val_split_index = int(np.floor(0.2 * (len(dataset_1) - test_split_index))) + test_split_index

test_idx = dataset_indices[:test_split_index]
val_idx = dataset_indices[test_split_index:val_split_index]
train_idx = dataset_indices[val_split_index:]

k = 20
trainset = Subset(dataset_1, train_idx[:int(np.floor(k * 1000))])
# trainset = Subset(dataset_1, train_idx)
# validset = Subset(dataset_2, val_idx)
validset = Subset(dataset_2, val_idx[:int(np.floor(k * 250))])
testset = Subset(dataset_3, test_idx)

training_loader = DataLoader(
    trainset,
    batch_size=8,
    shuffle=True,
    num_workers=16
)

validation_loader = DataLoader(
    validset,
    batch_size=8,
    shuffle=False,
    num_workers=8
)

test_loader = DataLoader(
    testset,
    batch_size=8,
    shuffle=True,
    num_workers=8
)

dataloaders = {'train': training_loader, 'val': validation_loader, 'test_loader': test_loader}

print("Full set size:", len(dataset_1))
print("Train set size: ", len(trainset))
print("Validation set size: ", len(validset))
print("Test set size: ", len(testset))

class_names = dataset_1.classes
print(class_names)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    model.to(device)

    total_step = len(training_loader)
    print("total_step:", total_step)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)
                # inputs.shape = (8, 5, 240, 240), labels.shape = (8,)
                optimizer.zero_grad()

                # forward
                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # outputs.shape = (8, 3), looks like below:
                    # tensor([[-0.9669, 0.1060, -0.0951],
                    #         [-0.8292, 1.1135, -0.4336],
                    #         [-0.8459, 0.3798, -0.5044],
                    #         [-0.7668, 0.5464, -0.1321],
                    #         [-0.6945, 0.4186, -0.3601],
                    #         [-0.6203, 0.5930, 0.0504],
                    #         [-0.6348, 0.5252, 0.0493],
                    #         [-1.1363, 0.7513, -0.1114]], device='cuda:0',
                    #        grad_fn= < AddmmBackward0 >)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
                epoch_loss = running_loss / len(trainset)
                epoch_acc = running_corrects.double() / len(trainset)
            else:
                epoch_loss = running_loss / len(validset)
                epoch_acc = running_corrects.double() / len(validset)

            print('{} Loss: {:.6f} Acc: {:.6f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_accuracy(net):
    confusion_matrix = torch.zeros(3, 3)
    correct = 0
    total = 0
    net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    print("confusion_matrix: ", confusion_matrix)
    print(confusion_matrix.diag() / confusion_matrix.sum(1))
    print("accuracy over all: ", correct / total)
    return correct / total


def visualize_model(model, validation_dataloader, num_images=9):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(validation_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 3, 3, images_so_far)
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                utils.showImages(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


model_ft = models.resnet18(pretrained=True)
# We have access to all the modules, layers, and their parameters, we can easily freeze them by setting
# the parameters'requires_grad flag to False. This would prevent calculating the gradients for these parameters
# in the backward step which in turn prevents the optimizer from updating them.
# for param in model_ft.parameters():
#     param.requires_grad = False
# replace the conv1 (keep its weights)
model_ft.conv1 = nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
# nn.init.xavier_uniform_(model_ft.conv1.weight)

# replace the output layer
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

criterion = nn.CrossEntropyLoss()

model_ft = model_ft.to(device)
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)
test_accuracy(model_ft)
# visualize_model(model_ft, validation_dataloader=validation_loader)
