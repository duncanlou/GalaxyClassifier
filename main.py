import copy
import time

import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# import from local project
from FitsImageFolder import FitsImageFolder
from mynetwork import CelestialClassficationNet
from optical_dataset import OptDataSet

print("torch version: ", torch.__version__)
batch_size = 16
# src_root_path = '/mnt/DataDisk/Duncan/sources'
src_root_path = '/mnt/DataDisk/Duncan/sources_for_Sean'
# src_root_path = '/mnt/DataDisk/Duncan/PS_small_set'
print(src_root_path)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

print("Reading catalogue...")

# Data augmentation and normalization for training
# Just normalization for validation
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=(-90, 90))

])
validation_transforms = transforms.Compose([
    transforms.ToTensor()
])

# 加载图像数据集，并在加载的时候对图像施加变换
whole_dataset = FitsImageFolder(root=src_root_path)
trainData = OptDataSet(whole_dataset, transform=train_transforms)
validationData = OptDataSet(whole_dataset, validation_transforms)
testData = OptDataSet(whole_dataset, validation_transforms)

# Create the index splits for training, validation and test
train_size = 0.8
num_train = len(whole_dataset)
indices = list(range(num_train))
split = int(np.floor(train_size * num_train))
split2 = int(np.floor((train_size + (1 - train_size) / 2) * num_train))
np.random.shuffle(indices)
train_idx, valid_idx, test_idx = indices[:split], indices[split:split2], indices[split2:]

trainset = Subset(trainData, indices=train_idx)
validset = Subset(validationData, indices=valid_idx)
testset = Subset(testData, indices=test_idx)

dataset = {'train': trainset, 'val': validset, 'test': testset}

training_loader = DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=16,
)

validation_loader = DataLoader(
    validset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8,
)

test_loader = DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8,
)

dataloaders = {'train': training_loader, 'val': validation_loader, 'test': test_loader}

print("Train set size: ", len(trainset))
print("Validation set size: ", len(validset))
print("Test set size: ", len(testset))

class_names = whole_dataset.classes
print(class_names)

training_epoch_loss = []
validation_epoch_loss = []
training_epoch_accuracy = []
validation_epoch_accuracy = []

note_files = "data/ps_source_training_notes"

with open(note_files, "w+") as f:
    f.write("Training_epoch_loss, Validation_epoch_loss, Training_epoch_accuracy, Validation_epoch_accuracy\n")


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
            for i, (inputs, labels, wise_asinh_mag) in enumerate(dataloaders[phase]):
                inputs, labels, wise_asinh_mag = inputs.to(device), labels.to(device), wise_asinh_mag.to(
                    device)  # inputs: [8, 5, 240, 240], labels:[8, 1], wise_asinh_mag:[8, 2]
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    image = inputs.float()
                    wise_asinh_mag = wise_asinh_mag.float()
                    outputs, extracted_features = model(image, wise_asinh_mag)
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

            epoch_loss = running_loss / len(dataset[phase])
            epoch_acc = running_corrects.double() / len(dataset[phase])

            print('{} Loss: {:.6f} Acc: {:.6f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':
                training_epoch_loss.append(epoch_loss)
                training_epoch_accuracy.append(epoch_acc)
            else:
                validation_epoch_loss.append(epoch_loss)
                validation_epoch_accuracy.append(epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        with open(note_files, 'a') as file:
            file.write(
                f"{training_epoch_loss[-1]}, {validation_epoch_loss[-1]}, {training_epoch_accuracy[-1]}, {validation_epoch_accuracy[-1]}\n")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Save the best model weights
    PATH = "data/trained_model/PS_classification_model_wts.pt"
    torch.save(best_model_wts, PATH)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_accuracy(net):
    confusion_matrix = torch.zeros(3, 3)
    correct = 0
    total = 0
    net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        for i, (inputs, labels, wise_asinh_mag) in enumerate(test_loader):
            inputs, labels, wise_asinh_mag = inputs.to(device), labels.to(device), wise_asinh_mag.to(device)
            image = inputs.float()
            wise_asinh_mag = wise_asinh_mag.float()
            outputs, extracted_features = net(image, wise_asinh_mag)
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
                showImages(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


model = CelestialClassficationNet().to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
test_accuracy(model_ft)

# visualize_model(model_ft, validation_dataloader=validation_loader)
