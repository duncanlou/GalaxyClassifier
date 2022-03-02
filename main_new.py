import copy
import os
import time

import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
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
def load_data(data_dir=src_root_path):
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(96),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(3),
            transforms.RandomRotation(degrees=45)]),
        'val': transforms.Compose([
            transforms.ToTensor()
        ])
    }

    # 加载图像数据集，并在加载的时候对图像施加变换
    dataset = FitsImageFolder(root=src_root_path)
    train_set_size = int(len(dataset) * 0.8)
    validation_set_size = int(len(dataset) * 0.1)
    test_set_size = len(dataset) - train_set_size - validation_set_size
    train_set, validation_set, test_set = random_split(dataset, [train_set_size, validation_set_size, test_set_size])
    train_set.dataset.transform = data_transforms['train']
    validation_set.dataset.transform = data_transforms['val']
    test_set.dataset.transform = data_transforms['val']

    print("Full set size:", len(dataset))
    print("Train set size: ", train_set_size)
    print("Validation set size: ", validation_set_size)
    print("Test set size: ", test_set_size)

    class_names = dataset.classes
    print(class_names)

    return train_set, validation_set, test_set


def train_model(config, checkpoint_dir=None, data_dir=None):
    since = time.time()

    model = models.resnext101_32x8d(pretrained=True)
    weight = model.conv1.weight.clone()
    model.conv1 = nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
    model.conv1 = nn.Sequential(nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=3), nn.BatchNorm2d(64),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    model.conv1 = nn.Sequential(model.conv1, nn.BatchNorm2d(64), nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.0001)
    num_epochs = 5

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    model.to(device)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    train_set, validation_set, test_set = load_data()

    training_loader = DataLoader(
        train_set,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8
    )

    validation_loader = DataLoader(
        validation_set,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8
    )

    dataloaders = {'train': training_loader, 'val': validation_loader}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
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
                exp_lr_scheduler.step()
                epoch_loss = running_loss / len(train_set)
                epoch_acc = running_corrects.double() / len(train_set)

            if phase == 'val':
                epoch_loss = running_loss / len(validation_set)
                epoch_acc = running_corrects.double() / len(validation_set)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=epoch_loss, accuracy=epoch_acc)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:6f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_accuracy(net, device='cpu'):
    _, _, test_set = load_data()
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


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=0.5):
    config = {
        # "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        # "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-6, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16, 32, 64, 128, 256]),
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])

    result = tune.run(
        tune.with_parameters(train_model, data_dir=src_root_path),
        resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=25, gpus_per_trial=1)
