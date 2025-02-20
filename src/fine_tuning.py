import os
import argparse
import numpy as np
import pandas as pd

from new_cnn_models import Cnn, dataset_partition, dataset_partition_augmented
from cnn_pytorch import EarlyStopper, testing, training, create_datasets

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import utils.consts as cons

import time
import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def parse_arguments(parser):
    parser.add_argument('--parent_dataset', default='fram', type=str)
    parser.add_argument('--child_dataset', default='steno_second', type=str)
    parser.add_argument('--noise_type', default='heterogeneous', type=str)
    parser.add_argument('--channels', default=1, type=int)
    parser.add_argument('--augmented', default=0, type=int)
    parser.add_argument('--n_cpus', default=15, type=int)
    parser.add_argument('--n_gpus', default=1, type=int)
    parser.add_argument('--n_workers', default=6, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--cuda', default=False, type=bool)
    return parser.parse_args()


if __name__ == "__main__":

    start_time = time.time()

    cmd_parser = argparse.ArgumentParser(description='train cnn')
    args = parse_arguments(cmd_parser)

    if not args.augmented:
        path = str(os.path.join(cons.PATH_PROJECT_SAVE_MODEL,
                                'model_cnn_classification_{}_{}.pt'.format(args.noise_type, args.parent_dataset)))
    else:
        path = str(os.path.join(cons.PATH_PROJECT_SAVE_MODEL,
                                'model_cnn_classification_{}_{}_ctgan.pt'.format(args.noise_type, args.parent_dataset)))
    models = torch.load(path)

    list_acc_values = []
    list_specificity_values = []
    list_recall_values = []
    list_auc_values = []

    for idx in range(len(cons.SEEDS)):
        model = models['trained_model'][idx]
        config = models['best_config'][idx]
        split_seed = cons.SEEDS[idx]

        if not args.augmented:
            path = os.path.join(cons.PATH_PROJECT_TAB_TO_IMAGE, f'image_{args.noise_type}_{args.child_dataset}')
            path_dataset_txt, path_dataset_png = create_datasets(path, args.child_dataset, args.noise_type)

            shape, train_subset, val_subset, test_subset, n_classes = dataset_partition(
                args.parent_dataset, args.noise_type, args.channels, config['min_shape'], path_dataset_txt, split_seed
            )

        else:
            file_name = 'train_{}_{}_{}_{}_seed_{}'.format(args.child_dataset, args.noise_type, args.type_sampling,
                                                           args.oversampler, split_seed)
            path = os.path.join(cons.PATH_PROJECT_TAB_TO_IMAGE, f'image_{args.noise_type}_{args.child_dataset}', 'ctgan',
                                file_name)
            train_path_dataset_txt, train_path_dataset_png = create_datasets(path, args.child_dataset, args.noise_type,
                                                                             file_name)
            file_name = 'test_{}_{}_{}_{}_seed_{}'.format(args.child_dataset, args.noise_type, args.type_sampling,
                                                          args.oversampler, split_seed)
            path = os.path.join(cons.PATH_PROJECT_TAB_TO_IMAGE, f'image_{args.noise_type}_{args.child_dataset}', 'ctgan',
                                file_name)
            test_path_dataset_txt, test_path_dataset_png = create_datasets(path, args.child_dataset, args.noise_type,
                                                                           file_name)
            path_dataset_txt, path_dataset_png = ([train_path_dataset_txt, test_path_dataset_txt],
                                                  [train_path_dataset_png, test_path_dataset_png])

            shape, train_subset, val_subset, test_subset = dataset_partition_augmented(
                args.parent_dataset, args.noise_type, args.channels, config['min_shape'], path_dataset_txt, split_seed
            )

        for param in model.parameters():
            param.requires_grad = True

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Descongelar fc1
        model.fc1 = nn.Linear(model.embedding_size, model.dense_units)
        model.relu_fc1 = nn.ReLU(inplace=False)
        model.fc1_bn = nn.BatchNorm1d(model.dense_units)
        model.dropout1 = nn.Dropout(model.dropout_rate, inplace=False)
        model.drop1_bn = nn.BatchNorm1d(model.dense_units)

        # Descongelar fc2
        model.fc2 = nn.Linear(model.fc2.in_features, 1)
        model.sig = nn.Sigmoid()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        criterion = nn.BCELoss()
        if config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        elif config['optimizer'] == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=config['learning_rate'])
        else:
            raise ValueError(f"Optimizer {config['optimizer']} not supported.")
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, min_lr=1e-20)
        early_stopper = EarlyStopper(patience=100, min_delta=0.0001, min_loss=0.05)

        n_epochs = 500
        batch_size = int(len(train_subset)/config['n_batches'])+1

        if args.n_workers > 0:
            train_loader = DataLoader(train_subset, batch_size=batch_size, num_workers=args.n_workers,
                                      persistent_workers=True, pin_memory=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, num_workers=args.n_workers,
                                    persistent_workers=True, pin_memory=True)
        else:
            train_loader = DataLoader(train_subset, batch_size=batch_size, pin_memory=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, pin_memory=True)

        val_loss, val_accuracy, best_trained_model, epoch_loss_list, epoch_loss_train = training(
            model, n_epochs, device, train_loader, val_loader, criterion, optimizer, early_stopper, scheduler
        )

        print(f"Best trial final validation loss: {val_loss}")
        print(f"Best trial final validation accuracy: {val_accuracy}")

        test_loader = DataLoader(test_subset, batch_size=batch_size)
        test_results = testing(best_trained_model, test_loader, device)

        list_acc_values.append(test_results[0])
        list_specificity_values.append(test_results[1])
        list_recall_values.append(test_results[2])
        list_auc_values.append(test_results[3])

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time} seconds")

    mean_std_specificity = np.mean(list_specificity_values), np.std(list_specificity_values)
    mean_std_accuracy = np.mean(list_acc_values), np.std(list_acc_values)
    mean_std_recall = np.mean(list_recall_values), np.std(list_recall_values)
    mean_std_auc = np.mean(list_auc_values), np.std(list_auc_values)

    print('accuracy:', mean_std_accuracy)
    print('specificity:', mean_std_specificity)
    print('recall:', mean_std_recall)
    print('AUC:', mean_std_auc)

    exp_name = '{}+{}+{}'.format(args.child_dataset, 'cnn', str(len(cons.SEEDS)))
    # exp_name = '{}+{}'.format(exp_name, 'fs') if args.flag_fs else exp_name

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

    if not args.augmented:
        csv_file_path = str(os.path.join(cons.PATH_PROJECT_METRICS,
                                         'metrics_fine_tuning_{}_{}.csv'.format(args.noise_type, args.child_dataset)))
    else:
        csv_file_path = str(os.path.join(cons.PATH_PROJECT_METRICS,
                                         'metrics_fine_tuning_{}_{}_ctgan.csv'.format(args.noise_type, args.child_dataset)))

    if os.path.exists(csv_file_path):
        try:
            df_metrics_classification = pd.read_csv(csv_file_path)
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
    df_metrics_classification.to_csv(csv_file_path, index=False)
