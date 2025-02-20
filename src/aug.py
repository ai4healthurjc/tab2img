import os
import pickle
import argparse
import numpy as np
import pandas as pd
from operator import itemgetter
from collections import Counter

import utils.consts as consts
from utils.loader import load_preprocessed_dataset, load_dataset_with_noise, normalize_dataframe

from ctgan import CTGAN

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

import time
import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)
list_conventional_oversamplers = ['rus', 'smoten']


def join_real_synthetic_data(x_real, x_synthetic, y_train, num_samples_class_maj, num_samples_class_min_syn,
                             id_label_min):
    x_train_resampled = np.concatenate((x_real, x_synthetic), axis=0)
    y_train_resampled = np.concatenate((y_train, np.reshape(np.full((num_samples_class_min_syn, 1), id_label_min), -1)))

    num_samples_class_min_real = y_train.shape[0]

    y_meta_class_min_real = np.reshape(np.full((num_samples_class_min_real, 1), 'Real'), -1)
    y_meta_class_min_syn = np.reshape(np.full((num_samples_class_min_syn, 1), 'Synthetic'), -1)
    y_meta_class_min_real_syn = np.concatenate((y_meta_class_min_real, y_meta_class_min_syn))
    y_meta_class_maj_real = np.reshape(np.full((num_samples_class_maj, 1), 'Real'), -1)

    y_meta_class_maj_min = np.concatenate((y_meta_class_maj_real, y_meta_class_min_real_syn))

    return x_train_resampled, y_train_resampled, y_meta_class_maj_min


def get_x_train_classes_and_ids(x_train, y_train, v_column_names, imbalance_ratio):
    c_items = Counter(y_train)
    id_label_min, num_samples_min = min(c_items.items(), key=itemgetter(1))
    id_label_maj, num_samples_maj = max(c_items.items(), key=itemgetter(1))

    df_x_train_with_label = pd.DataFrame(x_train, columns=v_column_names)
    df_x_train_with_label['label'] = list(y_train)

    df_x_train_class_min_with_label = df_x_train_with_label[df_x_train_with_label.loc[:, 'label'] == id_label_min]
    df_x_train_class_maj_with_label = df_x_train_with_label[df_x_train_with_label.loc[:, 'label'] == id_label_maj]

    df_x_train_class_min = df_x_train_class_min_with_label.iloc[:, :-1]
    df_x_train_class_maj = df_x_train_class_maj_with_label.iloc[:, :-1]

    y_train_class_min = df_x_train_class_min_with_label.loc[:, 'label']
    y_train_class_maj = df_x_train_class_maj_with_label.loc[:, 'label']

    num_synthetic_samples_required = int(imbalance_ratio * num_samples_maj)
    num_synthetic_samples = num_synthetic_samples_required - num_samples_min

    return (df_x_train_class_min, y_train_class_min, df_x_train_class_maj, y_train_class_maj, num_synthetic_samples,
            id_label_min, id_label_maj)


def parse_arguments(parser):
    parser.add_argument('--dataset', default='steno', type=str)
    parser.add_argument('--with_noise', default=1, type=int)
    parser.add_argument('--noise_type', default='heterogeneous', type=str)
    parser.add_argument('--categorical_encoding', default='count', type=str)
    parser.add_argument('--type_sampling', default='over', type=str)
    parser.add_argument('--oversampler', default='ctgan', type=str)
    parser.add_argument('--type_encoding', default='standard', type=str)
    parser.add_argument('--cuda', default=True, type=bool)
    parser.add_argument('--n_seeds', default=5, type=int)
    parser.add_argument('--n_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=50, type=int)
    return parser.parse_args()


if __name__ == "__main__":

    start_time = time.time()

    parser = argparse.ArgumentParser(description='Oversampling experiments')
    args = parse_arguments(parser)

    imbalance_ratio = 1

    list_acc_values = []
    list_specificity_values = []
    list_recall_values = []
    list_auc_values = []
    list_feature_importance = []

    if args.with_noise:
        generic_name = '{}_{}_{}_{}'.format(args.dataset, args.noise_type, args.type_sampling, args.oversampler)
    else:
        generic_name = '{}_{}_{}'.format(args.dataset, args.type_sampling, args.oversampler)

    for idx in consts.SEEDS:

        generic_name_partition = '{}_seed_{}'.format(generic_name, idx)
        logger.info('Experiment with: {}'.format(generic_name_partition))

        if args.with_noise:
            _, df_x_features, y_label = load_dataset_with_noise(args.dataset, args.noise_type)
        else:
            _, df_x_features, y_label = load_preprocessed_dataset(args.dataset, args.categorical_encoding)

        v_column_names = df_x_features.columns.values

        x_train, x_test, y_train, y_test = train_test_split(df_x_features, y_label,
                                                            test_size=0.2, random_state=idx)

        x_train, x_test = normalize_dataframe(x_train, x_test, args.type_encoding)

        (df_x_train_class_min_real, y_train_class_min_real, df_x_train_class_maj, y_train_class_maj, n_samples_syn,
         id_label_min, id_label_maj) = get_x_train_classes_and_ids(x_train, y_train, v_column_names, imbalance_ratio)

        num_samples_maj = df_x_train_class_maj.shape[0]
        num_samples_min = df_x_train_class_min_real.shape[0]

        if args.oversampler == 'ctgan':
            logger.info('Resampling with CTGAN - oversampling')
            oversampler_model = CTGAN(epochs=args.n_epochs,
                                      batch_size=args.batch_size,
                                      cuda=args.cuda,
                                      verbose=True
                                      )
            oversampler_model.fit(df_x_train_class_min_real, v_column_names)
        else:
            logger.info('Resampling with RUS - undersampling')
            oversampler_model = RandomUnderSampler(random_state=idx, sampling_strategy=imbalance_ratio)

        if args.oversampler in list_conventional_oversamplers:
            x_train_maj_resampled_min, y_train_maj_resampled_min = oversampler_model.fit_resample(x_train, y_train)
            y_meta_train_with_both_min_maj = y_train_maj_resampled_min
        else:
            df_x_train_class_min_syn = oversampler_model.sample(n_samples_syn)

            pickle.dump(oversampler_model, open(str(
                os.path.join(consts.PATH_PROJECT_SAVE_OVERSAMPLED, 'models', 'model_oversampler_{}.sav'.format(
                    generic_name_partition))), 'wb'))

            x_train_resampled_class_min, y_train_resampled_class_min, y_meta_train_with_both_min_maj = (
                join_real_synthetic_data(df_x_train_class_min_real.values, df_x_train_class_min_syn.values,
                                         y_train_class_min_real, num_samples_maj, n_samples_syn, id_label_min))

            x_train_maj_resampled_min = np.concatenate((df_x_train_class_maj.values, x_train_resampled_class_min),
                                                       axis=0)
            y_train_maj_resampled_min = np.concatenate((y_train_class_maj, y_train_resampled_class_min))

        if args.type_sampling == 'hybrid':
            logger.info('Training with hybrid approach')

            rus = RandomUnderSampler(random_state=idx)
            x_train_resampled, y_train_resampled = rus.fit_resample(x_train_maj_resampled_min,
                                                                    y_train_maj_resampled_min)
            v_indices_resampled = rus.sample_indices_
            y_label_real_syn_total_resampled = y_meta_train_with_both_min_maj[v_indices_resampled]

        elif args.type_sampling == 'over':
            logger.info('Training with oversampling approach')

            x_train_resampled = x_train_maj_resampled_min
            y_train_resampled = y_train_maj_resampled_min
            y_label_real_syn_total_resampled = y_meta_train_with_both_min_maj

        else:
            logger.info('Training with rus/smoten')
            x_train_resampled, y_train_resampled = oversampler_model.fit_resample(x_train, y_train)
            y_label_real_syn_total_resampled = [id_label_maj if label == 0 else id_label_min
                                                for label in y_train_resampled]

        logger.info('Resampled x_train dataset shape {}'.format(x_train_resampled.shape))
        logger.info('Resampled y_train dataset shape {}'.format(Counter(y_train_resampled)))

        if args.type_encoding == 'target':
            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train_resampled_scaled = scaler.transform(x_train_resampled)
            x_test_scaled = scaler.transform(x_test)
        else:
            x_train_resampled_scaled = x_train_resampled
            x_test_scaled = x_test

        if args.with_noise:
            path = consts.PATH_PROJECT_REPORTS_NOISY_DATASETS
        else:
            path = consts.PATH_PROJECT_DATA_PREPROCESSED

        df_train = pd.DataFrame(x_train_resampled_scaled, columns=v_column_names)
        df_train['label'] = y_train_resampled
        df_train.to_csv(str(os.path.join(path, args.oversampler, 'train_{}.csv'.format(generic_name_partition))),
                        index=False)
        df_test = pd.DataFrame(x_test_scaled, columns=v_column_names)
        df_test['label'] = list(y_test)
        df_test.to_csv(str(os.path.join(path, args.oversampler, 'test_{}.csv'.format(generic_name_partition))),
                       index=False)
