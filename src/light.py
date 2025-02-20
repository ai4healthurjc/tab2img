import os
import argparse
import shutil
import numpy as np

import util.consts as cons
import util.loader as loader

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import BinaryAccuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback

import time
import logging
import coloredlogs
from multiprocessing import cpu_count

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


# Prepare dataset
class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, split_seed, n_workers):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.split_seed = split_seed
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.transform = transforms.Compose([
            transforms.Resize(32, antialias=True),
            transforms.ToTensor(),
        ])
        self.train_data, self.val_data, self.test_data, self.predict_data = None, None, None, None

    def setup(self, stage: str):
        full_dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)

        labels = [full_dataset[label][1] for label in range(len(full_dataset))]
        idx = np.arange(len(labels))
        idx_train, idx_test, y_train, y_test = train_test_split(idx, labels, stratify=labels,
                                                                test_size=0.2, random_state=self.split_seed)
        idx_train, idx_val, y_train, y_val = train_test_split(idx_train, y_train, stratify=y_train,
                                                              test_size=0.2, random_state=self.split_seed)

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_data = Subset(full_dataset, idx_train)
            self.val_data = Subset(full_dataset, idx_val)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_data = Subset(full_dataset, idx_test)

        if stage == "predict":
            self.predict_data = Subset(full_dataset, idx_test)

    def train_dataloader(self):
        if self.n_workers > 0:
            return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers,
                              persistent_workers=True)
        else:
            return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        if self.n_workers > 0:
            return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers,
                              persistent_workers=True)
        else:
            return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        if self.n_workers > 0:
            return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers,
                              persistent_workers=True)
        else:
            return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        if self.n_workers > 0:
            return DataLoader(self.predict_data, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers,
                              persistent_workers=True)
        else:
            return DataLoader(self.predict_data, batch_size=self.batch_size, shuffle=False)


# Define CNN architecture
class CNN(pl.LightningModule):

    def __init__(self, config):
        """
        :param config: dictionary with all hyperparameters
        """
        super().__init__()
        self.accuracy = BinaryAccuracy()
        self.eval_loss = []
        self.eval_accuracy = []

        # ==========================
        # HYPERPARAMETERS
        # ==========================

        self.save_hyperparameters(config)
        self.learning_rate = config['learning_rate']
        self.optimizer = config['optimizer']
        self.filters = config['filters']
        self.kernel_size = config['kernel_size']
        self.pool_size = config['pool_size']
        self.dense_units = config['dense_units']
        self.dropout_rate = config['dropout_rate']

        # ==========================
        # CONVOLUTIONAL NETWORK
        # ==========================

        # first convolutional layer
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=self.filters, kernel_size=self.kernel_size)
        self.relu1 = nn.ReLU(inplace=True)
        self.cnn1_bn = nn.BatchNorm2d(self.filters)
        self.maxpool1 = nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size)
        self.max1_bn = nn.BatchNorm2d(self.filters)

        # second convolutional layer
        self.cnn2 = nn.Conv2d(in_channels=self.filters, out_channels=self.filters, kernel_size=self.kernel_size)
        self.relu2 = nn.ReLU(inplace=True)
        self.cnn2_bn = nn.BatchNorm2d(self.filters)
        self.maxpool2 = nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size)
        self.max2_bn = nn.BatchNorm2d(self.filters)

        # first fully connected layer
        self.embedding_size = self.filters * (
            int((int((32 - self.kernel_size + 1) / self.pool_size) - self.kernel_size + 1) / self.pool_size)) ** 2
        self.fc1 = nn.Linear(self.embedding_size, self.dense_units)
        self.fc1_bn = nn.BatchNorm1d(self.dense_units)
        self.dropout1 = nn.Dropout(self.dropout_rate, inplace=True)
        self.drop1_bn = nn.BatchNorm1d(self.dense_units)

        # second fully connected layer
        self.fc2 = nn.Linear(self.dense_units, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """
        :param x: image
        :return: prediction
        """
        # first convolutional layer: Conv2D + ReLU + BN
        x = self.cnn1_bn(self.relu1(self.cnn1(x)))
        # first max pooling
        x = self.max1_bn(self.maxpool1(x))
        # second convolutional layer: Conv2D + ReLU + BN
        x = self.cnn2_bn(self.relu2(self.cnn2(x)))
        # second max pooling
        x = self.max2_bn(self.maxpool2(x))
        # flatten output
        embed = x.view(x.size(0), -1)
        # first fully connected layer: fc + BN + dropout + BN
        x = self.drop1_bn(self.dropout1(self.fc1_bn(self.fc1(embed))))
        # second fully connected layer
        x = self.sig(self.fc2(x))
        return x

    # Training
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = torch.squeeze(self(x))
        y = y.to(dtype=torch.float)
        loss = F.binary_cross_entropy(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        self.log('ptl/train_loss', loss)
        self.log("ptl/train_accuracy", accuracy)
        return loss

    # Validation
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = torch.squeeze(self(x))
        y = y.to(dtype=torch.float)
        loss = F.binary_cross_entropy(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        self.eval_loss.append(loss)
        self.eval_accuracy.append(accuracy)
        # self.log('val_loss', loss)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.eval_loss).mean()
        avg_acc = torch.stack(self.eval_accuracy).mean()
        self.log("ptl/val_loss", avg_loss, sync_dist=True)
        self.log("ptl/val_accuracy", avg_acc, sync_dist=True)
        self.eval_loss.clear()
        self.eval_accuracy.clear()

    # Test
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = torch.squeeze(self(x))
        y = y.to(dtype=torch.float)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)

    # Prediction
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    # Optimizer
    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": ReduceLROnPlateau(optimizer, mode='min'),
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "ptl/val_loss",
                }
            }
        elif self.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": ReduceLROnPlateau(optimizer, mode='min'),
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "ptl/val_loss",
                }
            }
        else:
            raise ValueError(f"Optimizer {self.optimizer} not supported.")


# Create a folder to store data in image folder format
def create_dataset(dataset_id, noise_type, path_images, path_dataset):
    images_paths = loader.load_images(dataset_id, noise_type)
    _, _, labels = loader.load_dataset_with_noise(dataset_id, noise_type)

    rus = RandomUnderSampler(sampling_strategy='all', random_state=42)
    images_resampled, y_resampled = rus.fit_resample(images_paths, labels)
    print('Resampled dataset shape %s' % y_resampled.value_counts())

    images_resampled = [image[0].split('\\')[-1] for image in images_resampled]
    y_resampled = list(y_resampled)

    os.makedirs(path_dataset)
    for label in list(set(y_resampled)):
        os.makedirs(os.path.join(path_dataset, str(label)))
        for n, image_label in enumerate(y_resampled):
            if image_label == label:
                origin_path = os.path.join(path_images, images_resampled[n])
                final_path = os.path.join(os.path.join(path_dataset, str(label)), images_resampled[n])
                shutil.copy(str(origin_path), str(final_path))


def train_func(config, params):
    pl.seed_everything(42, workers=True)

    # Load data
    dm = DataModule(path_dataset, params['batch_size'], params['split_seed'], params['workers'])
    model = CNN(config)

    # Callbacks
    early_stop = EarlyStopping(monitor="ptl/val_loss", mode="min", min_delta=0.001)
    ray_callback = TuneReportCallback(
        {
            "loss": "ptl/val_loss",
            "mean_accuracy": "ptl/val_accuracy"
        },
        on="validation_end")

    # Train model
    trainer = pl.Trainer(max_epochs=1, num_sanity_val_steps=0, enable_model_summary=False, deterministic=True,
                         logger=TensorBoardLogger(save_dir=os.getcwd(), name="", version="."),
                         callbacks=[early_stop, ray_callback])

    trainer.fit(model=model, datamodule=dm)


# Arguments from the cmd to run the code
def parse_arguments(parser):
    parser.add_argument('--dataset', default='ionos', type=str)
    parser.add_argument('--noise_type', default='homogeneous', type=str)
    parser.add_argument('--type_padding', default='none', type=str)
    parser.add_argument('--padding', default=50, type=int)
    parser.add_argument('--type_interpolation', default='none', type=str)
    parser.add_argument('--interpolation', default=5, type=int)
    parser.add_argument('--n_seeds', default=5, type=int)
    parser.add_argument('--n_cpus', default=15, type=int)
    parser.add_argument('--n_gpus', default=1, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--cuda', default=False, type=bool)
    return parser.parse_args()


if __name__ == "__main__":

    start_time = time.time()

    # Read the parser
    cmd_parser = argparse.ArgumentParser(description='train cnn')
    args = parse_arguments(cmd_parser)

    # Create directory structure for loading data
    data_path = os.path.join(cons.PATH_PROJECT_TAB_TO_IMAGE, f'image_{args.noise_type}_{args.dataset}')
    path_images = os.path.join(data_path, 'data')
    path_dataset = os.path.join(data_path, 'dataset')

    if not os.path.exists(path_dataset):
        create_dataset(args.dataset, args.noise_type, path_images, path_dataset)

    # Establish number of workers
    n_procs = cpu_count()
    cpus = n_procs if args.n_cpus == -1 or args.n_cpus > n_procs else args.n_cpus
    logger.info('n_cpus for train_cnn: {}'.format(cpus))

    if torch.cuda.is_available():
        n_procs = 1
        if torch.cuda.device_count() > 1:
            n_procs = torch.cuda.device_count()
    gpus = n_procs if args.n_gpus == -1 or args.n_gpus > n_procs else args.n_gpus
    logger.info('n_gpus for train_cnn: {}'.format(gpus))

    pl.seed_everything(42, workers=True)

    for split_seed in range(args.n_seeds):
        search_space = {
            'filters': tune.grid_search([8, 16, 32, 64]),
            'kernel_size': tune.grid_search([3, 4, 5]),
            'pool_size': tune.grid_search([2, 3]),
            'optimizer': tune.grid_search(['adam', 'rmsprop']),
            'dense_units': tune.grid_search([32, 64, 128]),
            'learning_rate': tune.uniform(0.001, 0.01),
            'dropout_rate': tune.grid_search([0.1, 0.2, 0.4])
        }
        params = {
            'batch_size': args.batch_size,
            'split_seed': split_seed,
            'workers': 0
        }
        scheduler = ASHAScheduler(
            max_t=20, grace_period=1, reduction_factor=2
        )
        train_fn_with_parameters = tune.with_parameters(
            train_func, params=params
        )
        ray.shutdown()
        ray.init(num_cpus=cpus, num_gpus=gpus)
        tuner = tune.Tuner(
            tune.with_resources(train_fn_with_parameters, resources={'cpu': 3, 'gpu': 0.25}),
            param_space=search_space,
            tune_config=tune.TuneConfig(
                metric="loss",
                mode="min",
                num_samples=1,
                scheduler=scheduler,
            ),
        )
        results = tuner.fit()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time} seconds")
