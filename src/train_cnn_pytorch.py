from utils.loader import load_images
from pathlib import Path
import logging
import argparse
from utils.loader import load_dataset_with_noise
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.under_sampling import RandomUnderSampler
import cv2
import numpy as np
from multiprocessing import cpu_count
import coloredlogs
from sklearn.metrics import make_scorer
import time

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import Module
# from torch.nn import Conv2d
# from torch.nn import Linear
# from torch.nn import MaxPool2d
# from torch.nn import ReLU
# from torch.nn import LogSoftmax
from ray import tune
from ray.air import session
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader


class ConvNet(nn.Module):

    def __init__(self, l1=120, l2=84):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.batch1 = nn.BatchNorm2d(32, affine=True, track_running_stats=True)

        self.conv2 = nn.Conv2d(32, 156, 3)
        self.batch2 = nn.BatchNorm2d(56, affine=True, track_running_stats=True)

        self.conv3 = nn.Conv2d(56, 64, 3, stride=1, padding=1)
        self.batch3 = nn.BatchNorm2d(64, affine=True, track_running_stats=True)

        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.batch4 = nn.BatchNorm2d(64, affine=True, track_running_stats=True)

        # input size 32, 28, 28
        self.fc1 = nn.Linear(64 * 14 * 14, 4096)
        self.fc2 = nn.Linear(4096, 512)
        self.fc3 = nn.Linear(512, 64)
        self.fc4 = nn.Linear(64, 2)

        self.batch5 = nn.BatchNorm1d(512)
        self.batch6 = nn.BatchNorm1d(64)

        self.drop = nn.Dropout(p=.3)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.pool(self.batch1(x))

        x = F.relu(self.conv2(x), inplace=True)
        x = self.pool(self.batch2(x))
        x = self.drop(x)

        x = F.relu(self.conv3(x), inplace=True)
        x = self.pool(self.batch3(x))
        x = self.drop(x)

        x = F.relu(self.conv4(x), inplace=True)
        x = self.pool(self.batch4(x))

        x = x.view(-1, 64 * 14 * 14)

        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.batch5(x)
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = self.batch6(x)
        x = self.drop(x)
        x = self.fc4(x)

        return x


def train_cnn(config, trainset, testset):

    net = ConvNet(config["l1"], config["l2"])

    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    checkpoint = session.get_checkpoint()

    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        net.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    # trainset, testset = load_data(data_dir)
    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )

    validation_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )

    for epoch in range(start_epoch, 10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
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

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        session.report(
            {"loss": val_loss / val_steps, "accuracy": correct / total},
            checkpoint=checkpoint,
        )
    print("Finished Training")


def main_train(trainset, testset, num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    data_dir = os.path.abspath("./data")

    config = {
        "l1": tune.choice([2**i for i in range(9)]),
        "l2": tune.choice([2**i for i in range(9)]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16]),
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    result = tune.run(
        partial(train_cnn, trainset=trainset, testset=testset),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    best_trained_model = ConvNet(best_trial.config["l1"], best_trial.config["l2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    best_checkpoint_data = best_checkpoint.to_dict()
    best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


def test_accuracy(net, device="cpu"):
    trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def parse_arguments(parser):
    parser.add_argument('--dataset', default='ionos', type=str)
    parser.add_argument('--noise_type', default='homogeneous', type=str)
    parser.add_argument('--n_seeds', default=5, type=int)
    parser.add_argument('--n_jobs', default=4, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--cuda', default=False, type=bool)
    return parser.parse_args()


if __name__ == "__main__":

    start_time = time.time()

    parser = argparse.ArgumentParser(description='noise generator experiments')
    args = parse_arguments(parser)

    n_procs = cpu_count()
    n_jobs = n_procs if args.n_jobs == -1 or args.n_jobs > n_procs else args.n_jobs
    logger.info('n_jobs for train_cnn: {}'.format(n_jobs))

    images = load_images(args.dataset, args.noise_type)
    path_dataset, features, y_label = load_dataset_with_noise(args.dataset, args.noise_type)

    list_acc_values = []
    list_specificity_values = []
    list_recall_values = []
    list_auc_values = []

    rus = RandomUnderSampler(sampling_strategy='all', random_state=42)
    X_res, y_res = rus.fit_resample(images, y_label)
    print('Resampled dataset shape %s' % y_res.value_counts())

    imgs = []

    for i in range(len(X_res)):
        imgs.append((cv2.imread(X_res[i][0]))/255)

    imagenes_resize = []

    for imagen in range(len(imgs)):
        imagenes_resize.append(cv2.resize(imgs[imagen], (32, 32)))

    for idx in range(args.n_seeds):

        X_train, X_test, Y_train, Y_test = train_test_split(imagenes_resize, y_res, stratify=y_res, test_size=0.2, random_state=idx)
        # X_train, X_val, y_train, y_val = train_test_split(X_train, Y_train, stratify=Y_train, test_size=0.15, random_state=idx)

        X_train = np.array(X_train)
        X_test = np.array(X_test)
        # X_val = np.array(X_val)

        Y_train = np.array(Y_train)
        Y_test = np.array(Y_test)

        # Y_train = to_categorical(y_train, num_classes=len(np.unique(y_label)))
        # Y_test = to_categorical(Y_test, num_classes=len(np.unique(y_label)))
        # Y_val = to_categorical(y_val, num_classes=len(np.unique(y_label)))

        tensor_x_train = torch.Tensor(X_train)
        tensor_y_train = torch.Tensor(Y_train)

        tensor_x_test = torch.Tensor(X_test)
        tensor_y_test = torch.Tensor(Y_test)

        train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
        test_dataset = TensorDataset(tensor_x_test, tensor_y_test)

        main_train(trainset=train_dataset, testset=test_dataset, num_samples=10, max_num_epochs=10, gpus_per_trial=0)
