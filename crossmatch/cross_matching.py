import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from crossmatch.model_crossmatch import RadioOpticalCrossmatchModel
from mynetwork import CelestialClassficationNet
from ps_dataset import FitsImageSet
from utils import append_new_line
from xmatch_dataset import XMatchDataset

OPT_SOURCE_CLASSIFICATION_MODEL_PATH = "models/opt_classification_model_wts.pt"
RADIO_MODEL_PATH = "models/crossmatch_model_wts.pt"
VLASS_IMAGE_ROOT = "/mnt/DataDisk/Duncan/Pan-STARRS_Big_Cutouts/VLASS_training_data"
PS_IMAGE_ROOT = "/mnt/DataDisk/Duncan/Pan-STARRS_Big_Cutouts/PS_training_data"

print("torch version: ", torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# Training settings
num_classes = 2
batch_size = 8
epochs = 100
lr = 3e-5
gamma = 0.7
seed = 42

training_note_file = "training_notes/crossmatch_model_training_notes3.txt"
testing_note_file = "training_notes/Norris06_data_test_notes.txt"

append_new_line(training_note_file,
                "Training_epoch_loss, Validation_epoch_loss, Training_epoch_accuracy, Validation_epoch_accuracy")


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed)

opt_transform = transforms.Compose([
    transforms.ToTensor()])

radio_transform = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(216),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomInvert(),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(216),
    ])}


def create_testData():
    testDataset = FitsImageSet(radio_root_dir=VLASS_IMAGE_ROOT,
                               opt_root_dir=PS_IMAGE_ROOT,
                               ps_positive_samples_csv="../data/preprocessed_cat/PS_p_Norris06_samples.csv",
                               ps_negative_samples_csv="../data/preprocessed_cat/PS_n_Norris06_samples.csv")
    testData = XMatchDataset(testDataset, radio_transform=radio_transform['val'], opt_transform=opt_transform)
    indices = list(range(len(testData)))
    np.random.shuffle(indices)
    testset = Subset(testData, indices=indices)
    return testset


whole_dataset = FitsImageSet(radio_root_dir=VLASS_IMAGE_ROOT,
                             opt_root_dir=PS_IMAGE_ROOT,
                             ps_positive_samples_csv="../data/preprocessed_cat/PS_p_ROGUE_samples.csv",
                             ps_negative_samples_csv="../data/preprocessed_cat/PS_n_ROGUE_samples.csv")
trainData = XMatchDataset(whole_dataset, radio_transform=radio_transform['train'], opt_transform=opt_transform)
validationData = XMatchDataset(whole_dataset, radio_transform=radio_transform['val'], opt_transform=opt_transform)

# Create the index splits for training, validation and test
train_size = 0.8
num_train = len(whole_dataset)
indices = list(range(num_train))
split = int(np.floor(train_size * num_train))
np.random.shuffle(indices)
train_idx, valid_idx = indices[:split], indices[split:]

trainset = Subset(trainData, indices=train_idx)
validset = Subset(validationData, indices=valid_idx)
testset = create_testData()

dataset = {'train': trainset, 'val': validset, 'test': testset}

train_loader = DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8
)

validation_loader = DataLoader(
    validset,
    batch_size=1,
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    testset,
    batch_size=1,
    shuffle=False,
    num_workers=2
)

dataloaders = {'train': train_loader, 'val': validation_loader, 'test': test_loader}

print("Train set size: ", len(trainset))
print("Validation set size: ", len(validset))
print("Test set size: ", len(testset))

training_epoch_loss = []
validation_epoch_loss = []
training_epoch_accuracy = []
validation_epoch_accuracy = []


# Get one batch of training data
def test_one_batch():
    radio_img, ps_imgcube, cutout_pos_info, label = next(iter(train_loader))
    print(radio_img.shape)
    print(ps_imgcube.shape)
    print(cutout_pos_info)
    print(label)


def train_model(ps_model, model, criterion, optimizer, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    model.to(device)

    total_step = len(train_loader)
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

            for i, (
            radio_image, ps_imgcube, wise_asinh_mag, cutout_position_info, labels, ps_source_identity) in enumerate(
                    dataloaders[phase]):
                radio_image, ps_imgcube, wise_asinh_mag, cutout_position_info, labels = radio_image.to(
                    device), ps_imgcube.to(device), wise_asinh_mag.to(device), cutout_position_info.to(
                    device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    radio_image = radio_image.float()
                    cutout_position_info = cutout_position_info.float()
                    ps_imgcube = ps_imgcube.float()
                    wise_asinh_mag = wise_asinh_mag.float()

                    ps_model_outputs = ps_model(ps_imgcube, wise_asinh_mag)
                    # ps_source_class_probs = F.softmax(ps_model_outputs, dim=1)

                    # outputs = model(radio_image, ps_imgcube, ps_source_class_probs, cutout_position_info)
                    outputs = model(radio_image, ps_imgcube, ps_model_outputs, cutout_position_info)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * radio_image.size(0)
                running_corrects += torch.sum(preds == labels.data)

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

        append_new_line(training_note_file,
                        f"{training_epoch_loss[-1]}, {validation_epoch_loss[-1]}, {training_epoch_accuracy[-1]}, {validation_epoch_accuracy[-1]}")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Save the best model weights
    PATH = RADIO_MODEL_PATH
    torch.save(best_model_wts, PATH)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_accuracy(ps_model, net):
    confusion_matrix = torch.zeros(2, 2)
    correct = 0
    total = 0
    net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():  # ps_source_identity
        for i, (radio_image, ps_imgcube, wise_asinh_mag, cutout_position_info, labels, ps_source_identity) in enumerate(
                test_loader):
            radio_image, ps_imgcube, wise_asinh_mag, cutout_position_info, labels = radio_image.to(
                device), ps_imgcube.to(device), wise_asinh_mag.to(device), cutout_position_info.to(device), labels.to(
                device)

            radio_image = radio_image.float()
            cutout_position_info = cutout_position_info.float()
            ps_imgcube = ps_imgcube.float()
            wise_asinh_mag = wise_asinh_mag.float()

            ps_model_outputs = ps_model(ps_imgcube, wise_asinh_mag)
            # ps_source_class_probs = F.softmax(ps_model_outputs, dim=1)

            # outputs = net(radio_image, ps_imgcube, ps_source_class_probs, cutout_position_info)
            outputs = net(radio_image, ps_imgcube, ps_model_outputs, cutout_position_info)
            radio_host_probs = F.softmax(outputs, dim=1)
            radio_host_prob_list = radio_host_probs.tolist()
            host_prob = radio_host_prob_list[0]

            ps_id = ps_source_identity[0].item()
            ps_ra = ps_source_identity[1].item()
            ps_dec = ps_source_identity[2].item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pred = predicted.item()
            lab = labels.item()
            b = pred == lab
            append_new_line(testing_note_file, f"{ps_id}, {ps_ra}, {ps_dec}, {host_prob[0]}, {host_prob[1]}, {b}")

            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    print("confusion_matrix: ", confusion_matrix)
    print(confusion_matrix.diag() / confusion_matrix.sum(1))
    print("accuracy over all: ", correct / total)
    return correct / total


def set_ps_model():
    ps_source_classification_model = CelestialClassficationNet()
    ps_source_classification_model.load_state_dict(torch.load(OPT_SOURCE_CLASSIFICATION_MODEL_PATH))
    ps_source_classification_model.eval()
    ps_source_classification_model.to(device)
    return ps_source_classification_model


def set_radio_model():
    radio_model = RadioOpticalCrossmatchModel().to(device)
    radio_model.load_state_dict(torch.load(RADIO_MODEL_PATH))
    radio_model.eval()
    radio_model.to(device)
    return radio_model


if __name__ == '__main__':
    ps_model = set_ps_model()
    radio_model = set_radio_model()
    append_new_line(testing_note_file,
                    "PS source id, PS source ra, PS source dec, Host likelihood, Non-host likelihood, Prediction_result")
    test_accuracy(ps_model, radio_model)

    # x_model = RadioOpticalCrossmatchModel().to(device)
    #
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = Adam(x_model.parameters(), lr=0.001, weight_decay=0.0001)
    #
    # model_ft = train_model(ps_model, x_model, loss_fn, optimizer, num_epochs=10)
    #
    # append_new_line(testing_note_file, "PS source id, PS source ra, PS source dec, Host likelihood, Non-host likelihood, Prediction_result")
    #
    # test_accuracy(ps_model, model_ft)
