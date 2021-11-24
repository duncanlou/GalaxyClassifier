import os
import shutil
from typing import cast

import numpy as np

import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data import WeightedRandomSampler

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
np.random.seed(42)
torch.manual_seed(42)
src_root_path = os.path.join(os.getcwd(), "data/sources")  # galaxy: 7509; quasar: 738; star: 990


def make_datasets_sampler(dataset):
    targets = [dataset.dataset.targets[idx] for idx in dataset.indices]
    class_sample_count = np.unique(targets, return_counts=True)[1]
    weight = 1. / class_sample_count
    samples_weight = weight[targets]
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(
        weights=samples_weight,
        num_samples=len(samples_weight),
        replacement=True
    )
    return sampler


def load_data():
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

    return train_set, validation_set, test_set


# the accuracy on test set (not validation set!)
def test_accuracy(net, device=None):
    """Compute the accuracy for a model on a dataset using a GPU."""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)

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
    # net = GalaxyNet(config["l1"], config["l2"])
    net = GalaxyNet(120, 84)

    device = 'cpu'
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)
    print('Using {} device'.format(device))

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=config["lr"], weight_decay=1e-5)
    optimizer = optim.SGD(net.parameters(), lr=0.001, weight_decay=1e-5)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, validset = data
    train_sampler = make_datasets_sampler(trainset)
    valid_sampler = make_datasets_sampler(validset)

    trainloader = DataLoader(
        trainset,
        # batch_size=int(config["batch_size"]),
        batch_size=64,
        num_workers=2,
        sampler=train_sampler
    )


    animator = d2l.Animator(xlabel='epoch', xlim=[1, 10], legend=['train loss', 'train_acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(trainloader)
    valloader = DataLoader(
        validset,
        # batch_size=int(config["batch_size"]),
        batch_size=64,
        num_workers=2,
        sampler=valid_sampler
    )

    num_epochs = 10

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(trainloader, 0):
            timer.start()

            print("train batch index {}, GALAXY/QSO/STAR: {}/{}/{}".format(
                i,
                len(np.where(y.numpy() == 0)[0]),
                len(np.where(y.numpy() == 1)[0]),
                len(np.where(y.numpy() == 2)[0])
            ))

            X, y = X.to(device), y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            y_hat = net(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                metric.add(loss * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
                timer.stop()
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                    animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))

        # validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader):
            with torch.no_grad():
                images, labels = data

                print("validation batch index {}, GALAXY/QSO/STAR: {}/{}/{}".format(
                    i,
                    len(np.where(labels.numpy() == 0)[0]),
                    len(np.where(labels.numpy() == 1)[0]),
                    len(np.where(labels.numpy() == 2)[0])
                ))
                images, labels = images.to(device), labels.to(device)

                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        # with tune.checkpoint_dir(epoch) as checkpoint_dir:
        #     path = os.path.join(checkpoint_dir, "checkpoint")
        #     torch.save((net.state_dict(), optimizer.state_dict()), path)

        val_acc = correct / total
        average_loss = val_loss / val_steps
        tune.report(loss=average_loss, accuracy=val_acc)

        animator.add(epoch + 1, (None, None, val_acc))

    print(f'loss {train_l:.3f}, train acc {train_acc * 100:.5f}%, ' f'test acc {val_acc * 100:.5f}%')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec ' f'on {str(device)}')

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
    train_set, validation_set, test_set = load_data()
    train_source_classifier(data=(train_set, validation_set))
