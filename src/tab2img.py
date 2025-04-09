import os
import pandas as pd
import argparse
import math
from utils.loader import load_dataset_with_noise, load_dataset_augmented, load_preprocessed_dataset
from igtd.igtd_functions import min_max_transform, table_to_image, select_features_by_variation
from multiprocessing import cpu_count
import logging
import coloredlogs
import utils.consts as cons

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def tab_to_image_fnc(data, path):
    num_features = len(data.columns)
    num_row = int(math.sqrt(num_features))
    num_col = int(num_features / num_row)

    num = num_row * num_col  # Number of features to be included for analysis, which is also the total number of pixels in image representation
    save_image_size = 3  # Size of pictures (in inches) saved during the execution of IGTD algorithm.

    # Select features with large variations across samples

    # Perform min-max transformation so that the maximum and minimum values of every feature become 1 and 0, respectively. CUIDADOOOO REVISAR
    norm_data = min_max_transform(data.values)
    norm_data = pd.DataFrame(norm_data, columns=data.columns, index=data.index)
    table_to_image(norm_data.iloc[:, :num_row*num_col],
                   [num_row, num_col],
                   args.corr,
                   args.distances,
                   save_image_size,
                   args.max_step,
                   args.val_step,
                   path,
                   args.err,
                   args.interpretability
                   )


def parse_arguments(parser):
    parser.add_argument('--dataset', default='hepatitis', type=str)
    parser.add_argument('--noise_type', default='preprocessed', type=str)
    parser.add_argument('--categorical_encoding', default='count', type=str)
    parser.add_argument('--augmented', default=0, type=int)
    parser.add_argument('--with_noise', default=0, type=int)
    parser.add_argument('--type_sampling', default='over', type=str)
    parser.add_argument('--oversampler', default='ctgan', type=str)
    parser.add_argument('--corr', default='mixed', type=str)
    parser.add_argument('--n_boots', default=100, type=int)
    parser.add_argument('--n_jobs', default=50, type=int)
    parser.add_argument('--max_step', default=2000, type=int)
    parser.add_argument('--val_step', default=300, type=int)
    parser.add_argument('--fs', default='relief', type=str)
    parser.add_argument('--type_aggregation', default='mean', type=str)
    parser.add_argument('--err', default='squared', type=str)
    parser.add_argument('--distances', default='Gower', type=str)
    parser.add_argument('--interpretability', default=1, type=int)
    return parser.parse_args()


parser = argparse.ArgumentParser(description='noise generator experiments')
args = parse_arguments(parser)

n_procs = cpu_count()
n_jobs = n_procs if args.n_jobs == -1 or args.n_jobs > n_procs else args.n_jobs
logger.info('n_jobs for tab_to_image_fs: {}'.format(n_jobs))

result_dir = os.path.join(cons.PATH_PROJECT_REPORTS_TAB_TO_IMAGE,
                          'image_{}_{}'.format(args.noise_type, args.dataset))
os.makedirs(name=result_dir, exist_ok=True)

if args.augmented:
    os.makedirs(name=os.path.join(result_dir, 'ctgan'), exist_ok=True)
    for idx in cons.SEEDS:
        file_name, X_train_scaled, X_test_scaled, Y_train, Y_test = load_dataset_augmented(
            args.with_noise, idx, args.dataset, args.type_sampling, args.oversampler, args.noise_type)

        train_subset_path = os.path.join(os.path.join(result_dir, 'ctgan'), f'train_{file_name}')
        os.makedirs(name=train_subset_path, exist_ok=True)
        tab_to_image_fnc(X_train_scaled, train_subset_path)

        test_subset_path = os.path.join(os.path.join(result_dir, 'ctgan'), f'test_{file_name}')
        os.makedirs(name=test_subset_path, exist_ok=True)
        tab_to_image_fnc(X_test_scaled, test_subset_path)

else:
    if args.noise_type=='preprocessed':
        path_dataset, data, y_label = load_preprocessed_dataset(args.dataset, args.categorical_encoding)
    else:
        path_dataset, data, y_label = load_dataset_with_noise(args.dataset, args.noise_type)
    tab_to_image_fnc(data, result_dir)
