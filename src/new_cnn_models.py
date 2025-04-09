import os
import cv2
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import utils.loader as loader
from torchvision import transforms, datasets
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from ray import train
import pandas as pd


class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, loader=None):
        super().__init__(root, transform=transform, loader=loader)

    def __getitem__(self, index):
        image, label = super().__getitem__(index)

        img_name = self.imgs[index][0]

        return image, label, img_name
import os
import cv2
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import utils.loader as loader
from torchvision import transforms, datasets
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from ray import train
import pandas as pd


class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, loader=None):
        super().__init__(root, transform=transform, loader=loader)

    def __getitem__(self, index):
        image, label = super().__getitem__(index)

        img_name = self.imgs[index][0]

        return image, label, img_name

def partition_augmentation(path_images, channels, img_transforms, split_seed):
    if channels == 1:
        train_dataset = datasets.ImageFolder(root=path_images[0], transform=img_transforms,
                                             loader=img_loader_one_channel)
        test_dataset = datasets.ImageFolder(root=path_images[1], transform=img_transforms,
                                            loader=img_loader_one_channel)
    else:
        train_dataset = datasets.ImageFolder(root=path_images[0], transform=img_transforms, loader=cv2.imread)
        test_dataset = datasets.ImageFolder(root=path_images[1], transform=img_transforms, loader=cv2.imread)

    labels = [train_dataset[label][1] for label in range(len(train_dataset))]
    idx = np.arange(len(labels))

    rus = RandomUnderSampler(sampling_strategy='all', random_state=split_seed)
    idx_train_re, y_train_re = rus.fit_resample(idx.reshape(-1, 1), labels)
    print('Resampled label train dataset shape %s')
    print(pd.DataFrame(y_train_re).value_counts())
    idx_train, idx_val, y_train, y_val = train_test_split(idx_train_re, y_train_re, stratify=y_train_re,
                                                          test_size=0.2, random_state=split_seed)
    train_subset = Subset(train_dataset, idx_train)
    val_subset = Subset(train_dataset, idx_val)
    return train_subset, val_subset, test_dataset, len(set(labels))


def partition(gradcam, channels, path_images, img_transforms, split_seed):
    if gradcam:
        if channels == 1:
            full_dataset = CustomImageFolder(root=path_images[1], transform=img_transforms,
                                          loader=img_loader_one_channel)
        else:
            full_dataset = CustomImageFolder(root=path_images, transform=img_transforms,
                                         loader=cv2.imread)
    else:
        if channels == 1:
            full_dataset = datasets.ImageFolder(root=path_images, transform=img_transforms,
                                                loader=img_loader_one_channel)
        else:
            full_dataset = datasets.ImageFolder(root=path_images, transform=img_transforms, loader=cv2.imread)

    labels = [full_dataset[label][1] for label in range(len(full_dataset))]
    idx = np.arange(len(labels))
    idx_train, idx_test, y_train, y_test = train_test_split(idx, labels, stratify=labels,
                                                            test_size=0.2, random_state=split_seed)

    rus = RandomUnderSampler(sampling_strategy='all', random_state=split_seed)
    idx_train_re, y_train_re = rus.fit_resample(idx_train.reshape(-1, 1), y_train)
    print('Resampled label train dataset shape %s')
    print(pd.DataFrame(y_train_re).value_counts())
    print('Test label dataset shape %s')
    print(pd.DataFrame(y_test).value_counts())

    idx_train, idx_val, y_train, y_val = train_test_split(idx_train_re, y_train_re, stratify=y_train_re,
                                                          test_size=0.2, random_state=split_seed)
    train_subset = Subset(full_dataset, idx_train.reshape(-1))
    val_subset = Subset(full_dataset, idx_val.reshape(-1))
    test_subset = Subset(full_dataset, idx_test.reshape(-1))

    return train_subset, val_subset, test_subset, len(set(labels))


def dataset_partition(dataset, noise_type, channels, min_shape, path_images, split_seed, interpretability=False,
                      gradcam=False, augmentation=False):
    shape = determine_size(dataset, noise_type, min_shape, interpretability=interpretability)
    img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(shape, antialias=True),
    ])
    if augmentation:
        train_subset, val_subset, test_subset, n_labels =partition_augmentation(path_images, channels, img_transforms, split_seed)
    else:
        train_subset, val_subset, test_subset, n_labels = partition(gradcam, channels, path_images, img_transforms, split_seed)



    return shape, train_subset, val_subset, test_subset, n_labels



def load_one_channel_image(dataset_id, noise_type, other_path=None, interpretability=False):
    if interpretability:
        images_paths = loader.load_images_interpretability(dataset_id, noise_type)
        imgage_path_first_image = images_paths[0][0].split('data_interpretability')[0]
        imgage_path_first_image = os.path.join(imgage_path_first_image, 'dataset_from_png_interpretability/0/')

    else:
        images_paths = loader.load_images(dataset_id, noise_type, other_path)
        imgage_path_first_image = images_paths[0][0].split('data')[0]
        imgage_path_first_image = os.path.join(imgage_path_first_image, 'dataset_from_txt/0/')


    first_image_name = os.listdir(imgage_path_first_image)[0]
    imgage_path_first_image = os.path.join(imgage_path_first_image, first_image_name)
    image = img_loader_one_channel(imgage_path_first_image)
    return image


def determine_size(dataset_id, noise_type, min_shape, other_path=None, interpretability=False):
    image = load_one_channel_image(dataset_id, noise_type, other_path, interpretability)
    shape = image.shape
    og_shape = shape
    while min(shape) < min_shape:
        shape = tuple(x + y for x, y in zip(shape, og_shape))
    return shape


def img_loader_one_channel(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def find_best_model(config, data):
    if data['augmented']:
        shape, train_subset, val_subset,  _ ,num_classes = dataset_partition(
            data['dataset'], data['noise_type'], data['channels'], config['min_shape'], data['path_images'],
            data['split_seed'], augmentation=True
        )
    else:
        shape, train_subset, val_subset, _ ,num_classes= dataset_partition(
            data['dataset'], data['noise_type'], data['channels'], config['min_shape'], data['path_images'],
            data['split_seed']
        )

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = Cnn(shape[0], shape[1], data['channels'], config, num_classes=num_classes)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.BCELoss()
    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    elif config['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config['learning_rate'])
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported.")

    batch_size = int(len(train_subset)/config['n_batches']) + 1

    train_loader = DataLoader(train_subset, batch_size=batch_size, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, pin_memory=True)

    for epoch in range(5):
        model.train()
        for i, images in enumerate(train_loader):
            inputs, labels = images
            inputs, labels = inputs.to(dtype=torch.float).to(device), labels.to(dtype=torch.float).to(device)
            optimizer.zero_grad()
            if len(labels) >1:
                outputs = torch.squeeze(model(inputs))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            else:
                continue

        model.eval()
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, images in enumerate(val_loader):

            with torch.no_grad():
                inputs, labels = images
                inputs, labels = inputs.to(dtype=torch.float).to(device), labels.to(dtype=torch.float).to(device)
                if len(labels) > 1:
                    outputs = torch.squeeze(model(inputs))
                    predicted = (outputs.data >= 0.5)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    loss = criterion(outputs, labels)
                    val_loss += loss.cpu().numpy()
                    val_steps += 1
                else:
                    continue

    epoch_loss = val_loss / val_steps
    epoch_accuracy = correct / total

    train.report({"loss": epoch_loss, "accuracy": epoch_accuracy})

#
# class Cnn(nn.Module):
#     def __init__(self, in_size_height, in_size_width, in_channels, config):
#         """
#         :param in_size_height: initial height of the image
#         :param in_size_width: initial width of the image
#         :param config: hyperparameter configuration
#         """
#         super(Cnn, self).__init__()
#         self.filters = config['filters']
#         self.kernel_size = config['kernel_size']
#         self.pool_size = config['pool_size']
#         self.dense_units = config['dense_units']
#         self.dropout_rate = config['dropout_rate']
#
#         # first convolutional layer
#         self.cnn1 = nn.Conv2d(in_channels=in_channels, out_channels=self.filters, kernel_size=self.kernel_size)
#         self.relu1 = nn.ReLU()
#         self.cnn1bn = nn.BatchNorm2d(self.filters)
#         self.maxpool1 = nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size)
#         self.max1bn = nn.BatchNorm2d(self.filters)
#
#         # second convolutional layer
#         self.cnn2 = nn.Conv2d(in_channels=self.filters, out_channels=self.filters, kernel_size=self.kernel_size)
#         self.relu2 = nn.ReLU()
#         self.cnn2bn = nn.BatchNorm2d(self.filters)
#         self.maxpool2 = nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size)
#         self.max2bn = nn.BatchNorm2d(self.filters)
#         self.dropout = nn.Dropout(self.dropout_rate)
#
#         # first fully connected layer
#         self.embedding_size = (self.filters * (
#          int((int((in_size_height - self.kernel_size + 1) / self.pool_size) - self.kernel_size + 1) / self.pool_size)) *
#          (int((int((in_size_width - self.kernel_size + 1) / self.pool_size) - self.kernel_size + 1) / self.pool_size)))
#         self.fc1 = nn.Linear(self.embedding_size, self.dense_units)
#         self.relufc1 = nn.ReLU()
#         self.fc1bn = nn.BatchNorm1d(self.dense_units)
#         self.dropout1 = nn.Dropout(self.dropout_rate,)
#         self.drop1bn = nn.BatchNorm1d(self.dense_units)
#
#         # second fully connected layer
#         self.fc2 = nn.Linear(self.dense_units, 1)
#         self.sig = nn.Sigmoid()
#
#     def forward(self, x):
#         """
#         :param x: image
#         :return: prediction
#         """
#         x = self.cnn1(x)
#         x = self.cnn1bn(x)
#         x = self.relu1(x)
#         x = self.maxpool1(x)
#
#         x = self.cnn2(x)
#         x = self.cnn2bn(x)
#         x = self.maxpool2(x)
#
#         embed = x.view(x.size(0), -1)
#
#         x = self.fc1(embed)
#         x = self.fc1bn(x)
#         x = self.relufc1(x)
#         x = self.dropout1(x)
#         x = self.fc2(x)
#         x = self.sig(x)
#         return x

class Cnn(nn.Module):
    def __init__(self, in_size_height, in_size_width, in_channels, config, num_classes):
        """
        :param in_size_height: initial height of the image
        :param in_size_width: initial width of the image
        :param config: hyperparameter configuration
        :param num_classes: number of output classes (2 for binary, >2 for multiclass)
        """

        super(Cnn, self).__init__()
        self.filters = config['filters']
        self.kernel_size = config['kernel_size']
        self.pool_size = config['pool_size']
        self.dense_units = config['dense_units']
        self.dropout_rate = config['dropout_rate']
        self.num_classes = num_classes

        # first convolutional layer
        self.cnn1 = nn.Conv2d(in_channels=in_channels, out_channels=self.filters, kernel_size=self.kernel_size)
        self.relu1 = nn.ReLU()
        self.cnn1bn = nn.BatchNorm2d(self.filters)
        self.maxpool1 = nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size)

        # second convolutional layer
        self.cnn2 = nn.Conv2d(in_channels=self.filters, out_channels=self.filters, kernel_size=self.kernel_size)
        self.relu2 = nn.ReLU()
        self.cnn2bn = nn.BatchNorm2d(self.filters)
        self.maxpool2 = nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size)
        self.dropout = nn.Dropout(self.dropout_rate)

        # first fully connected layer
        self.embedding_size = (self.filters * (
         int((int((in_size_height - self.kernel_size + 1) / self.pool_size) - self.kernel_size + 1) / self.pool_size)) *
         (int((int((in_size_width - self.kernel_size + 1) / self.pool_size) - self.kernel_size + 1) / self.pool_size)))
        self.fc1 = nn.Linear(self.embedding_size, self.dense_units)
        self.relufc1 = nn.ReLU()
        self.fc1bn = nn.BatchNorm1d(self.dense_units)
        self.dropout1 = nn.Dropout(self.dropout_rate,)
        self.drop1bn = nn.BatchNorm1d(self.dense_units)

        # second fully connected layer
        self.fc2 = nn.Linear(self.dense_units, 1)
        self.sig = nn.Sigmoid()

        if num_classes == 2:
            self.fc2 = nn.Linear(self.dense_units, 1)
            self.activation = nn.Sigmoid()  # Sigmoid para clasificación binaria
            # Para clasificación binaria (una neurona)
        else:
            self.fc2 = nn.Linear(self.dense_units, num_classes)
            self.activation = None  # Softmax para clasificación multiclasess# Para clasificación multiclase (num_classes neuronas)

    def forward(self, x):
        """
        :param x: image
        :return: prediction
        """
        x = self.cnn1(x)
        x = self.relu1(x)
        x = self.cnn1bn(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = self.relu2(x)
        x = self.cnn2bn(x)
        x = self.maxpool2(x)

        embed = x.view(x.size(0), -1)

        x = self.fc1(embed)
        x = self.relufc1(x)
        x = self.fc1bn(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        if self.num_classes == 2:
            x = self.activation(x)
        return x

