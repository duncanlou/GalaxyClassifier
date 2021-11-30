import os
import shutil
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split, Subset


import torchvision.utils
from torchvision import transforms

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# import from local project

from FitsFolder import FitsFolder
from models.GalaxyNet import GalaxyNet
import d2l

print("torch version: ", torch.__version__)
writer = SummaryWriter('runs/source_classifier')

src_root_path = os.path.join(os.getcwd(), "data/sources")  # galaxy: 7243; quasar: 1014; star: 982

tfs = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(224),
])

# 加载图像数据集，并在加载的时候对图像施加变换
full_dataset = FitsFolder(root=src_root_path, transform=tfs)

train_set_size = int(len(full_dataset) * 0.8)
validation_set_size = int(len(full_dataset) * 0.1)
test_set_size = len(full_dataset) - train_set_size - validation_set_size

train_set, validation_set, test_set = random_split(full_dataset,
                                                   [train_set_size, validation_set_size, test_set_size])

print("Full set size:", len(full_dataset))
print("Train set size: ", train_set_size)
print("Validation set size: ", validation_set_size)
print("Test set size: ", test_set_size)


# the accuracy on test set (not validation set!)
def test_accuracy(net, device=None):
    """Compute the accuracy for a model on a dataset using a GPU."""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device


    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=4, shuffle=False, num_workers=2)

    with torch.no_grad():
        for X, y in testloader:
            if isinstance(X, list):  # Required for BERT Fine-tuning
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)

            metric.add(d2l.accuracy(net(X), y), y.numel())

    return metric[0] / metric[1]


def train_source_classifier(data, checkpoint_dir=None):
    model = GalaxyNet(120, 84)


    device = 'cpu'
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    print('Using {} device'.format(device))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-5)

    # if checkpoint_dir:
    #     model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
    #     net.load_state_dict(model_state)
    #     optimizer.load_state_dict(optimizer_state)

    trainset, validset = data

    trainloader = DataLoader(
        trainset,
        # batch_size=int(config["batch_size"]),
        batch_size=64,
        num_workers=2,
        shuffle=True
    )

    dataiter = iter(trainloader)
    images, labels = dataiter.__next__()



    valloader = DataLoader(
        validset,
        # batch_size=int(config["batch_size"]),
        batch_size=64,
        num_workers=2,
        shuffle=False
    )

    num_epochs = 10
    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    total_step = len(trainloader)

    for epoch in range(1, num_epochs + 1):  # loop over the dataset multiple times
        running_loss = 0.0
        # scheduler.step(epoch)
        correct = 0
        total = 0
        print(f"Epoch {epoch}\n")
        for batch_idx, (data_, target_) in enumerate(trainloader):
            data_, target_ = data_.to(device), target_.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(data_)
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred == target_).item()
            total += target_.size(0)
            if (batch_idx % 10) == 0:
                print("Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}".format(epoch, 10, batch_idx, total_step, loss.item()))
        train_acc.append(100 * correct / total)
        train_loss.append(running_loss / total_step)
        print(f"\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f} %")

        # validation loss
        batch_loss = 0
        total_t = 0
        correct_t = 0
        with torch.no_grad():
            model.eval()
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                labels_hat = model(images)
                loss_t = criterion(labels_hat, labels)
                batch_loss += loss_t.item()
                _, predicted = torch.max(labels_hat, dim=1)
                correct_t += torch.sum(predicted == labels).item()
                total_t += labels.size(0)

            val_acc.append(f"{100 * correct_t / total_t } %")
            val_loss.append(batch_loss / len(valloader))
            network_learned = batch_loss < valid_loss_min
            print(f"validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f} %\n")
            if network_learned:
                valid_loss_min = batch_loss
                torch.save(model.state_dict(), "model_classification_tutorial.pt")
                print('Detected network improvement, saving current model')
        model.train()

        # with tune.checkpoint_dir(epoch) as checkpoint_dir:
        #     path = os.path.join(checkpoint_dir, "checkpoint")
        #     torch.save((net.state_dict(), optimizer.state_dict()), path)
        # print("correct", correct)
        # print("total", total)
        # val_acc = correct / total
        # average_loss = val_loss / val_steps
        # print(f"Epoch{epoch: } the average loss is {average_loss}, the accuracy is {val_acc}")
        # tune.report(loss=average_loss, accuracy=val_acc)

        # animator.add(epoch + 1, (None, None, val_acc))

    # print(f'loss {train_l:.3f}, train acc {train_acc * 100:.5f}%, ' f'test acc {val_acc * 100:.5f}%')
    # print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec ' f'on {str(device)}')

    print('Finished Training')


# def main(num_samples=10, max_num_epochs=10, gpus_per_trial=0.5):
#     config = {
#         # the l1 and l2 parameters should be powers of 2 between 4 and 256, so either 4, 8, 16, 32, 64, 128, or 256
#         "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
#         "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
#         # The lr (learning rate) should be uniformly sampled between 0.0001 and 0.1
#         "lr": tune.loguniform(1e-4, 1e-1),
#         # the batch size is a choice between 8, 16, 32, 64, 128, 256 and 512.
#         "batch_size": tune.choice([8, 16, 32, 64, 128, 256]),
#     }
#
#     # ASHAScheduler will terminate bad performing trials early
#     scheduler = ASHAScheduler(
#         metric="loss",
#         mode="min",
#         max_t=max_num_epochs,
#         grace_period=1,
#         reduction_factor=2
#     )
#
#     reporter = CLIReporter(
#         # parametre_columns=["l1", "l2", "lr", "batch_size"],
#         metric_columns=["loss", "accuracy", "training_iteration"]
#     )
#
#     result = tune.run(
#         tune.with_parameters(train_source_classifier, data=(train_set, validation_set)),
#         resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
#         config=config,
#         num_samples=num_samples,
#         scheduler=scheduler,
#         progress_reporter=reporter
#     )
#     best_trial = result.get_best_trial("loss", "min", "last")
#     print("Best trial config: {}".format(best_trial.config))
#     print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
#     print("Best trial final validation accuracy: {}%".format(best_trial.last_result["accuracy"] * 100))
#
#     best_trained_model = GalaxyNet(best_trial.config["l1"], best_trial.config["l2"])
#     device = 'cpu'
#     if torch.cuda.is_available():
#         device = "cuda:0"
#         if gpus_per_trial > 1:
#             best_trained_model = nn.DataParallel(best_trained_model)
#     best_trained_model.to(device)
#
#     best_checkpoint_dir = best_trial.checkpoint.value
#     model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
#     best_trained_model.load_state_dict(model_state)
#
#     test_acc = test_accuracy(best_trained_model, device)
#     print("Best trial test set accuracy: {}%".format(test_acc * 100))


if __name__ == "__main__":
    # main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)
    train_source_classifier(data=(train_set, validation_set))
