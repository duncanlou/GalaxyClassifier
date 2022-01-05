import os

import numpy as np
# from torch and its affiliated packages
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
# from ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
# from matplotlib
import matplotlib.pyplot as plt

from utils import showImages

plt.ion()  # interactive mode
# import from local project
from FitsImageFolder import FitsImageFolder
from models.protoNet import GalaxyNet

print("torch version: ", torch.__version__)

src_root_path = os.path.join(os.getcwd(), "data/sources")


def load_data(data_dir=src_root_path):
    tfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90)
    ])

    # 加载图像数据集，并在加载的时候对图像施加变换
    dataset = FitsImageFolder(root=src_root_path, transform=tfs)
    train_set_size = int(len(dataset) * 0.8)
    test_set_size = len(dataset) - train_set_size
    train_set, test_set = random_split(dataset, [train_set_size, test_set_size])

    print("Full set size:", len(dataset))
    print("Train set size: ", train_set_size)
    print("Test set size: ", test_set_size)

    return train_set, test_set


num_epochs = 5





def train_loop(config, checkpoint_dir=None, data_dir=None):
    model = GalaxyNet(config["l1"], config["l2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], weight_decay=1e-5)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    train_set, test_set = load_data(data_dir)
    test_abs = int(len(train_set) * 0.8)
    train_subset, val_subset = random_split(train_set, [test_abs, len(train_set) - test_abs])

    training_loader = DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8
    )

    validation_loader = DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8
    )

    total_step = len(training_loader)
    print("total_step:", total_step)
    for epoch in range(num_epochs):
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(training_loader):
            if batch_idx == 0:
                showImages(images[0])
            images, labels = images.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if batch_idx % 20 == 0:
                print("[%d, %5d] loss: %.5f" % (epoch + 1, batch_idx + 1,
                                                running_loss / 20))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(validation_loader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)

    print("Finished Training")


def test_accuracy(net, device='cpu'):
    trainset, testset = load_data()

    testloader = DataLoader(testset, batch_size=8, shuffle=False, num_workers=2)

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


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=0.5):
    load_data(src_root_path)
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16, 32, 64, 128]),
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
        tune.with_parameters(train_loop, data_dir=src_root_path),
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

    best_trained_model = GalaxyNet(best_trial.config["l1"], best_trial.config["l2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)
    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0.5)
