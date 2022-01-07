import copy
import os
import time

import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models

# import from local project
import utils
from FitsImageFolder import FitsImageFolder

print("torch version: ", torch.__version__)

src_root_path = os.path.join(os.getcwd(), "data/sources")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(96),
        transforms.RandomResizedCrop(80, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=45)]),
    'val': transforms.Compose([
        transforms.ToTensor(),
    ])
}

# 加载图像数据集，并在加载的时候对图像施加变换
dataset = FitsImageFolder(root=src_root_path, transform=data_transforms['train'])
train_set_size = int(len(dataset) * 0.8)
validation_set_size = int(len(dataset) * 0.1)
test_set_size = len(dataset) - train_set_size - validation_set_size
train_set, validation_set, test_set = random_split(dataset, [train_set_size, validation_set_size, test_set_size])

training_loader = DataLoader(
    train_set,
    batch_size=64,
    shuffle=True,
    num_workers=8
)

validation_loader = DataLoader(
    validation_set,
    batch_size=64,
    shuffle=True,
    num_workers=8
)

dataloaders = {'train': training_loader, 'val': validation_loader}

print("Full set size:", len(dataset))
print("Train set size: ", train_set_size)
print("Validation set size: ", validation_set_size)
print("Test set size: ", test_set_size)

class_names = dataset.classes
print(class_names)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    model.to(device)

    total_step = len(training_loader)
    print("total_step:", total_step)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
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
                epoch_loss = running_loss / train_set_size
                epoch_acc = running_corrects.double() / train_set_size
            else:
                epoch_loss = running_loss / validation_set_size
                epoch_acc = running_corrects.double() / validation_set_size

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


def test_accuracy(net, device='cpu'):
    testloader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

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
                ax.aixs('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                utils.showImages(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


model_ft = models.resnext101_32x8d(pretrained=True)

weight = model_ft.conv1.weight.clone()

model_ft.conv1 = nn.Sequential(nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=3), nn.BatchNorm2d(64),
                               nn.ReLU(),
                               nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
visualize_model(model_ft, validation_dataloader=validation_loader)
