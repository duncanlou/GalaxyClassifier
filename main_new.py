import copy
import os

import numpy as np
import torch
import torch.optim as optim
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# import from local project
from FitsImageFolder import FitsImageFolder
from mynetwork import CelestialClassficationNet

print("torch version: ", torch.__version__)
torch.cuda.empty_cache()

src_root_path = '/mnt/DataDisk/Duncan/sources_short_version'
print(src_root_path)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Data augmentation and normalization for training
# Just normalization for validation
def load_data(data_dir=src_root_path):
    data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    # 加载图像数据集，并在加载的时候对图像施加变换
    dataset_1 = FitsImageFolder(root=src_root_path, transform=data_transforms)
    dataset_2 = FitsImageFolder(root=src_root_path, transform=data_transforms)
    dataset_3 = FitsImageFolder(root=src_root_path, transform=data_transforms)

    dataset_indices = list(range(len(dataset_1)))
    np.random.shuffle(dataset_indices)

    test_split_index = int(np.floor(0.1 * len(dataset_1)))
    val_split_index = int(np.floor(0.1 * (len(dataset_1) - test_split_index))) + test_split_index

    test_idx = dataset_indices[:test_split_index]
    val_idx = dataset_indices[test_split_index:val_split_index]
    train_idx = dataset_indices[val_split_index:]

    trainset = Subset(dataset_1, train_idx)
    validset = Subset(dataset_2, val_idx)
    testset = Subset(dataset_3, test_idx)

    return trainset, validset, testset


def train_model(config, checkpoint_dir=None, data_dir=None):
    model = CelestialClassficationNet(config["l1"], config["l2"])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.0001)
    num_epochs = 10

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

    total_step = len(training_loader)
    print("total_step:", total_step)

    for epoch in range(num_epochs):
        epoch_steps = 0
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            printing_running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    image = inputs[:, 0:5, :, :]
                    flux_layer = inputs[:, -1, :, :]
                    wise_asinh_mag = flux_layer[:, 0, 0:2]

                    image = image.float()
                    wise_asinh_mag = wise_asinh_mag.float()
                    outputs = model(image, wise_asinh_mag)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                printing_running_loss += loss.item()

                epoch_steps += 1
                if i % 20 == 19:  # print every 2000 mini-batches
                    print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                    printing_running_loss / epoch_steps))
                    printing_running_loss = 0.0

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

    print('Best val Acc: {:6f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_accuracy(net, device='cpu'):
    _, _, test_set = load_data()
    testloader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=2)

    confusion_matrix = torch.zeros(3, 3)
    correct = 0
    total = 0
    net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            image = inputs[:, 0:5, :, :]
            flux_layer = inputs[:, -1, :, :]
            wise_asinh_mag = flux_layer[:, 0, 0:2]

            image = image.float()
            wise_asinh_mag = wise_asinh_mag.float()
            outputs = net(image, wise_asinh_mag)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    print("confusion_matrix: ", confusion_matrix)
    print(confusion_matrix.diag() / confusion_matrix.sum(1))
    print("accuracy over all: ", correct / total)
    return correct / total


# def visualize_model(model, validation_dataloader, num_images=9):
#     was_training = model.training
#     model.eval()
#     images_so_far = 0
#     fig = plt.figure()
#
#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(validation_dataloader):
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#
#             for j in range(inputs.size()[0]):
#                 images_so_far += 1
#                 ax = plt.subplot(num_images // 3, 3, images_so_far)
#                 ax.aixs('off')
#                 ax.set_title('predicted: {}'.format(class_names[preds[j]]))
#                 utils.showImages(inputs.cpu().data[j])
#
#                 if images_so_far == num_images:
#                     model.train(mode=was_training)
#                     return
#         model.train(mode=was_training)


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=0.5):
    data_dir = os.path.abspath("./data")
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-6, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16, 32]),
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
        tune.with_parameters(train_model, data_dir=data_dir),
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

    best_trained_model = CelestialClassficationNet(best_trial.config["l1"], best_trial.config["l2"])
    best_trained_model.to(device)
    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=1)
