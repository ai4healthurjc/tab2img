import os
import argparse
import copy
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
from utils.dataset_creation import create_datasets
from new_cnn_models import Cnn, find_best_model, dataset_partition
import utils.consts as cons
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
import ray
from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search.hyperopt import HyperOptSearch
import time
import logging
import coloredlogs
from multiprocessing import cpu_count

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.01, min_loss=0.1):
        """
        :param patience: times loss get worse before stopping
        :param min_delta: how much loss get worse
        """
        self.patience = patience
        self.min_delta = min_delta
        self.min_loss = min_loss
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        """
        :param validation_loss: validation loss of an epoch
        :return:
            - True: stop training
            - False: continue training
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            if self.min_validation_loss < self.min_loss:
                return True

        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True

        else:
            return False


def compute_classification_performance(y_true: np.array, y_pred: np.array) -> (float, float, float, float):
    """
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: accuracy, specificity, recall and roc_auc
    """
    performance = classification_report(y_true, y_pred)
    print(performance)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc_val = accuracy_score(y_true, y_pred)
    specificity_val = tn / (tn + fp)
    recall_val = recall_score(y_true, y_pred)
    roc_auc_val = roc_auc_score(y_true, y_pred)

    return acc_val, specificity_val, recall_val, roc_auc_val


def training(model, n_epochs, device, train_loader, val_loader, criterion, optimizer, early_stopper, scheduler,num_classes):
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    minimum_loss = np.inf
    best_loss, best_acc, best_model_state = None, None, None
    best_model = copy.deepcopy(model)

    epoch_loss_list = []
    epoch_loss_list_train = []
    for epoch in range(n_epochs):

        model.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(dtype=torch.float).to(device), labels.to(dtype=torch.float).to(device)
            optimizer.zero_grad()
            if len(labels) > 1:
                outputs = torch.squeeze(model(inputs))
                if num_classes == 2:
                    labels = labels.squeeze()
                    loss = criterion(outputs, labels)
                else:
                    labels = labels.long()
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                epoch_loss_list_train.append(loss)
            else:
                continue

        model.eval()
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        valid_loop = tqdm(val_loader, leave=True)
        valid_loop.set_description(f"Epoch [{epoch + 1}/{n_epochs}] Validation")
        for i, data in enumerate(valid_loop):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(dtype=torch.float).to(device), labels.to(dtype=torch.float).to(device)

                if len(labels) >1:
                    outputs = torch.squeeze(model(inputs))
                    if num_classes == 2:
                        predicted = (outputs.data >= 0.5)
                    else:
                        predicted = torch.argmax(outputs.data, dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    loss2 = criterion(outputs, labels)
                    val_loss += loss2.cpu().numpy()
                    val_steps += 1

                    valid_loop.set_postfix(valid_loss=val_loss/val_steps)
                else:
                    continue

        epoch_loss = val_loss / val_steps
        epoch_accuracy = correct / total

        if epoch_loss < minimum_loss:
            minimum_loss = epoch_loss
            best_loss = epoch_loss
            best_acc = epoch_accuracy
            best_model_state = model.state_dict()

        scheduler.step(epoch_loss)
        epoch_loss_list.append(epoch_loss)
        if early_stopper.early_stop(epoch_loss):
            break

    best_model.load_state_dict(best_model_state)

    return best_loss, best_acc, best_model, epoch_loss_list, epoch_loss_list_train


def testing(model, test_loader, num_classes, device='cpu'):
    """
    :param model: trained model
    :param device: device use to test the model
    :param test_loader: data to test
    :return: average test loss and accuracy
    """
    predict_label = []
    real_label = []

    model.eval()

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(dtype=torch.float).to(device), labels.to(dtype=torch.float).to(device)
            if len(labels) > 1:
                outputs = torch.squeeze(model(images))
                if num_classes==2:
                    try:
                        prediction = np.array([int(np.round(i.cpu(), 0)) for i in outputs.data])
                    except TypeError:
                        prediction = np.array([int(np.round(outputs.data.cpu(), 0))])
                else:
                    prediction = torch.argmax(outputs, dim=1).cpu().numpy()
                labels = np.array([int(i.cpu()) for i in labels])
                predict_label.extend(prediction)
                real_label.extend(labels)
            else:
                continue

    results = compute_classification_performance(real_label, predict_label)

    return results


def parse_arguments(parser):
    parser.add_argument('--dataset', default='hepatitis', type=str)
    parser.add_argument('--noise_type', default='preprocessed', type=str)
    parser.add_argument('--channels', default=1, type=int)
    parser.add_argument('--augmented', default=0, type=int)
    parser.add_argument('--type_sampling', default='over', type=str)
    parser.add_argument('--oversampler', default='ctgan', type=str)
    parser.add_argument('--type_padding', default='none', type=str)
    parser.add_argument('--padding', default=50, type=int)
    parser.add_argument('--type_interpolation', default='none', type=str)
    parser.add_argument('--interpolation', default=5, type=int)
    parser.add_argument('--n_cpus', default=15, type=int)
    parser.add_argument('--n_gpus', default=1, type=int)
    parser.add_argument('--n_workers', default=12, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    return parser.parse_args()


if __name__ == "__main__":

    start_time = time.time()

    cmd_parser = argparse.ArgumentParser(description='train cnn')
    args = parse_arguments(cmd_parser)

    n_procs = cpu_count()
    cpus = n_procs if args.n_cpus == -1 or args.n_cpus > n_procs else args.n_cpus
    logger.info('n_cpus for train cnn: {}'.format(cpus))

    if torch.cuda.is_available():
        n_procs = 1
        if torch.cuda.device_count() > 1:
            n_procs = torch.cuda.device_count()
    gpus = n_procs if args.n_gpus == -1 or args.n_gpus > n_procs else args.n_gpus
    logger.info('n_gpus for train cnn: {}'.format(gpus))

    for split_seed in cons.SEEDS:

        if not args.augmented:
            path = os.path.join(cons.PATH_PROJECT_TAB_TO_IMAGE, f'image_{args.noise_type}_{args.dataset}')
            path_dataset_txt, path_dataset_png = create_datasets(path, args.dataset, args.noise_type)

        else:
            file_name = 'train_{}_{}_{}_{}_seed_{}'.format(args.dataset, args.noise_type, args.type_sampling,
                                                           args.oversampler)
            path = os.path.join(cons.PATH_PROJECT_TAB_TO_IMAGE, f'image_{args.noise_type}_{args.dataset}', 'ctgan',
                                file_name)
            train_path_dataset_txt, train_path_dataset_png = create_datasets(path, args.dataset, args.noise_type,
                                                                             file_name)

            file_name = 'test_{}_{}_{}_{}_seed_{}'.format(args.dataset, args.noise_type, args.type_sampling,
                                                          args.oversampler)
            path = os.path.join(cons.PATH_PROJECT_TAB_TO_IMAGE, f'image_{args.noise_type}_{args.dataset}', 'ctgan',
                                file_name)
            test_path_dataset_txt, test_path_dataset_png = create_datasets(path, args.dataset, args.noise_type,
                                                                           file_name)

            path_dataset_txt, path_dataset_png = ([train_path_dataset_txt, test_path_dataset_txt],
                                                  [train_path_dataset_png, test_path_dataset_png])

        if not args.augmented:
            result_file_path = str(os.path.join(cons.PATH_PROJECT_SAVE_MODEL,
                                                'result_cnn_classification_{}_{}_{}.pt'.format(
                                                    args.noise_type, args.dataset, split_seed)))
        else:
            result_file_path = str(os.path.join(cons.PATH_PROJECT_SAVE_MODEL,
                                                'result_cnn_classification_{}_{}_{}_ctgan.pt'.format(
                                                    args.noise_type, args.dataset, split_seed)))

        if not os.path.exists(result_file_path):

            config = {
                'filters': tune.choice([8, 16, 32, 64]),
                'kernel_size': tune.choice([3, 4, 5]),
                'pool_size': tune.choice([2, 3]),
                'optimizer': tune.choice(['adam', 'rmsprop']),
                'dense_units': tune.choice([32, 64, 128]),
                'learning_rate': tune.uniform(0.0001, 0.01),
                'dropout_rate': tune.choice([0.1, 0.2, 0.4]),
                'min_shape': tune.choice([25, 35, 45]),
                'n_batches': tune.choice([10,20,30])
            }

            ray.shutdown()

            runtime_env = {
                'env_vars': {
                    "RAY_memory_usage_threshold": "0.7",
                }
            }
            ray.init(runtime_env=runtime_env,num_cpus=cpus, num_gpus=gpus) #, local_mode=True

            params = {
                'dataset': args.dataset,
                'noise_type': args.noise_type,
                'path_images': path_dataset_txt,
                'channels': args.channels,
                'split_seed': split_seed,
                'augmented': args.augmented
            }

            """
            # Scheduler but with no seed
            scheduler = ASHAScheduler(
                metric="loss",
                mode="min",
                max_t=10,
                grace_period=1,
                reduction_factor=2
            )
            
            # Scheduler but with seed
            algo = BasicVariantGenerator(
                random_state=42
            )
            """
            # Define search algorithm
            algo = TuneBOHB(seed=42)
            algo = ConcurrencyLimiter(algo, max_concurrent=10)

            scheduler = HyperBandForBOHB(
                time_attr="training_iteration",
                max_t=10000,
                reduction_factor=4,
                stop_last_trials=False)
            """
            algo = HyperOptSearch(random_state_seed=42)
            algo = ConcurrencyLimiter(algo, max_concurrent=8)
            """
            """
            algo = OptunaSearch(seed=42)
            algo = ConcurrencyLimiter(algo, max_concurrent=8)
            """

            train_fn_with_params = tune.with_parameters(find_best_model, data=params)
            result = tune.run(
                train_fn_with_params,
                resources_per_trial={"cpu": 1, "gpu": gpus/cpus},
                config=config,
                metric="loss",
                mode="min",
                num_samples=30,
                search_alg=algo,
                scheduler=scheduler
            )

            torch.save(result, result_file_path)
            ray.shutdown()

    if not args.augmented:
        metrics_file_path = str(os.path.join(cons.PATH_PROJECT_METRICS,
                                'metrics_classification_cnn_{}_{}.csv'.format(args.noise_type,
                                                                              args.dataset)))
    else:
        metrics_file_path = str(os.path.join(cons.PATH_PROJECT_METRICS,
                                'metrics_classification_cnn_{}_{}_ctgan.csv'.format(args.noise_type,
                                                                                    args.dataset)))

    if not os.path.exists(metrics_file_path):
        list_acc_values = []
        list_specificity_values = []
        list_recall_values = []
        list_auc_values = []

        models = {'split_seed': [], 'best_config': [], 'trained_model': []}

        for split_seed in cons.SEEDS:

            if not args.augmented:
                result_file_path = str(os.path.join(cons.PATH_PROJECT_SAVE_MODEL,
                                                    'result_cnn_classification_{}_{}_{}.pt'.format(
                                                        args.noise_type, args.dataset, split_seed)))
            else:
                result_file_path = str(os.path.join(cons.PATH_PROJECT_SAVE_MODEL,
                                                    'result_cnn_classification_{}_{}_{}_ctgan.pt'.format(
                                                        args.noise_type, args.dataset, split_seed)))

            result = torch.load(result_file_path)
            best_trial = result.get_best_trial("loss", "min", "last")
            best_config = best_trial.config
            print(f"Best trial config: {best_config}")

            if args.augmented:
                shape, train_subset, val_subset, test_subset, n_classes = dataset_partition(
                    args.dataset, args.noise_type, args.channels, best_config['min_shape'], path_dataset_txt, split_seed, augmentation=True
                )
            else:
                shape, train_subset, val_subset, test_subset, n_classes = dataset_partition(
                    args.dataset, args.noise_type, args.channels, best_config['min_shape'], path_dataset_txt, split_seed
                )

            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            best_model = Cnn(shape[0], shape[1], args.channels, best_config, n_classes)
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda:0"
            best_model.to(device)

            criterion = nn.BCELoss()
            if best_config['optimizer'] == 'adam':
                optimizer = torch.optim.Adam(best_model.parameters(), lr=best_config['learning_rate'])
            elif best_config['optimizer'] == 'rmsprop':
                optimizer = torch.optim.RMSprop(best_model.parameters(), lr=best_config['learning_rate'])
            else:
                raise ValueError(f"Optimizer {best_config['optimizer']} not supported.")
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-20)
            early_stopper = EarlyStopper(patience=30, min_delta=0.0001, min_loss=0.20)

            n_epochs = 500
            batch_size = int(len(train_subset)/best_config['n_batches']) + 1

            if args.n_workers > 0:
                train_loader = DataLoader(train_subset, batch_size=batch_size, num_workers=args.n_workers,
                                          persistent_workers=True, pin_memory=True)
                val_loader = DataLoader(val_subset, batch_size=batch_size, num_workers=args.n_workers,
                                        persistent_workers=True, pin_memory=True)
            else:
                train_loader = DataLoader(train_subset, batch_size=batch_size, pin_memory=True)
                val_loader = DataLoader(val_subset, batch_size=batch_size, pin_memory=True)

            val_loss, val_accuracy, best_trained_model, epoch_loss_list, epoch_loss_train = training(
                best_model, n_epochs, device, train_loader, val_loader, criterion, optimizer, early_stopper, scheduler, n_classes
            )

            print(f"Best trial final validation loss: {val_loss}")
            print(f"Best trial final validation accuracy: {val_accuracy}")

            models['split_seed'].append(split_seed)
            models['best_config'].append(best_config)
            models['trained_model'].append(best_trained_model)

            test_loader = DataLoader(test_subset, batch_size=batch_size)
            test_results = testing(best_trained_model, test_loader,n_classes, device)

            list_acc_values.append(test_results[0])
            list_specificity_values.append(test_results[1])
            list_recall_values.append(test_results[2])
            list_auc_values.append(test_results[3])
   
        mean_std_specificity = np.mean(list_specificity_values), np.std(list_specificity_values)
        mean_std_accuracy = np.mean(list_acc_values), np.std(list_acc_values)
        mean_std_recall = np.mean(list_recall_values), np.std(list_recall_values)
        mean_std_auc = np.mean(list_auc_values), np.std(list_auc_values)

        print('accuracy:', mean_std_accuracy)
        print('specificity:', mean_std_specificity)
        print('recall:', mean_std_recall)
        print('AUC:', mean_std_auc)

        exp_name = '{}+{}+{}'.format(args.dataset, 'cnn', str(len(cons.SEEDS)))

        new_row_auc = {'model': exp_name,
                       'eval_metric': 'auc',
                       'mean': mean_std_auc[0],
                       'std': mean_std_auc[1]}

        new_row_sensitivity = {'model': exp_name,
                               'eval_metric': 'sensitivity',
                               'mean': mean_std_recall[0],
                               'std': mean_std_recall[1]}

        new_row_specificity = {'model': exp_name,
                               'eval_metric': 'specificity',
                               'mean': mean_std_specificity[0],
                               'std': mean_std_specificity[1]}

        new_row_accuracy = {
            'model': exp_name,
            'eval_metric': 'accuracy',
            'mean': mean_std_accuracy[0],
            'std': mean_std_accuracy[1]
        }

        if os.path.exists(metrics_file_path):
            try:
                df_metrics_classification = pd.read_csv(metrics_file_path)
            except pd.errors.EmptyDataError:
                df_metrics_classification = pd.DataFrame()
        else:
            df_metrics_classification = pd.DataFrame()

        df_metrics_classification = pd.concat([df_metrics_classification, pd.DataFrame([new_row_auc])],
                                              ignore_index=True)
        df_metrics_classification = pd.concat([df_metrics_classification, pd.DataFrame([new_row_sensitivity])],
                                              ignore_index=True)
        df_metrics_classification = pd.concat([df_metrics_classification, pd.DataFrame([new_row_specificity])],
                                              ignore_index=True)
        df_metrics_classification = pd.concat([df_metrics_classification, pd.DataFrame([new_row_accuracy])],
                                              ignore_index=True)
        df_metrics_classification.to_csv(metrics_file_path, index=False)

        if not args.augmented:
            model_file_path = str(os.path.join(cons.PATH_PROJECT_SAVE_MODEL,
                                               'model_cnn_classification_{}_{}.pt'.format(
                                                   args.noise_type, args.dataset)))
        else:
            model_file_path = str(os.path.join(cons.PATH_PROJECT_SAVE_MODEL,
                                               'model_cnn_classification_{}_{}_ctgan.pt'.format(
                                                   args.noise_type, args.dataset)))

        torch.save(models, model_file_path)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time} seconds")
