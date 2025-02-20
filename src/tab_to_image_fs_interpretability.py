import numpy as np
import pandas as pd
import argparse
import math
from pathlib import Path
from multiprocessing import cpu_count
import coloredlogs
import logging
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

from utils.loader import load_dataset_with_noise
from utils.IGTDTransformer_Interpretability import IGTD_Transformer
from utils.preprocessing import identify_feature_type
import features.ensemble_fs as fs
import utils.consts as consts

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def parse_arguments(parser):
    parser.add_argument('--dataset', default='hepatitis', type=str)
    parser.add_argument('--n_boots', default=100, type=int)
    parser.add_argument('--n_jobs', default=4, type=int)
    parser.add_argument('--fs', default='relief', type=str)
    parser.add_argument('--agg_func', default='mean', type=str)
    parser.add_argument('--type_corr', default='pearson', type=str)
    parser.add_argument('--noise_type', default='heterogeneous', type=str)

    return parser.parse_args()


cmd_parser = argparse.ArgumentParser(description='noise generator experiments')
args = parse_arguments(cmd_parser)

n_procs = cpu_count()
n_jobs = n_procs if args.n_jobs == -1 or args.n_jobs > n_procs else args.n_jobs
logger.info('n_jobs for tab_to_image_fs: {}'.format(n_jobs))

path_dataset, features, y_label = load_dataset_with_noise(args.dataset, args.noise_type)
dataset, categorical, numeric = identify_feature_type(features, categorical_threshold=0.05, impute_missing=False)

num_features = len(features.columns)  # Total number of characteristics in data set
print(features.columns)
# Calculate the number of rows and columns in the image based on the number of features
print(num_features)
num_row = int(math.sqrt(num_features))
num_col = int(math.ceil(num_features / num_row))

if num_features == (num_row * num_col):

    logger.info(
        f"Your dataset {args.dataset} has a total of {num_features} features: Images will be generated with a size - ({num_col},{num_row})")

    # for i in range(features.shape[0]):
    csv_file_path = str(Path.joinpath(consts.PATH_PROJECT_TAB_TO_IMAGE, f'image_{args.noise_type}_{args.dataset}'))
    best_noisy_data_generator = IGTD_Transformer(num_row=num_row,
                                                 num_col=num_col,
                                                 save_image_size=1,
                                                 max_step=1000,
                                                 val_step=200,
                                                 result_dir=csv_file_path,
                                                 numericas=numeric,
                                                 categoricas=categorical,
                                                 error='abs',
                                                 save_path=csv_file_path
                                                 )
    noisy_data = best_noisy_data_generator.transform(features)
    # noisy_data_with_labels = pd.concat([noisy_data, pd.Series(y_label, name='y_label')], axis=1)

    # noisy_data_with_labels.to_csv(csv_file_path, index=False)
else:
    logger.info(
        f"With the number of variables in your dataset with noise {args.dataset} it is not possible to generate a valid image size. Feature Selection techniques will be applied to generate a valid size")

    rus = RandomUnderSampler(sampling_strategy='all', random_state=42)
    X_res, y_res = rus.fit_resample(features, y_label)
    print('Resampled dataset shape %s' % y_res.value_counts())
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(X_res)

    y_res = np.array(y_res)
    v_feature_names = features.columns
    print(v_feature_names)
    features_scaled = pd.DataFrame(features_scaled)
    features_scaled.columns = v_feature_names

    list_vars_categorical, list_vars_numerical = fs.get_categorical_numerical_names(features_scaled)
    Z_selected, Z_scores = fs.compute_zmatrix_bootstrap(features_scaled,
                                                        y_res,
                                                        args.fs,
                                                        v_feature_names,
                                                        list_vars_categorical,
                                                        list_vars_numerical,
                                                        M=args.n_boots,
                                                        n_jobs=n_jobs)

    df_ensemble_voting_sorted, df_ensemble_mean_sorted = fs.run_ensemble_agg(features_scaled,
                                                                             Z_selected,
                                                                             Z_scores,
                                                                             args.agg_func)

    print(df_ensemble_voting_sorted)

    df = df_ensemble_voting_sorted.sort_values(by="score", ascending=False)

    num_features = df.shape[0]

    current_total = num_features
    current_rows = int(math.sqrt(current_total))
    current_columns = int(math.ceil(current_total / current_rows))

    while current_rows * current_columns > current_total:
        current_total -= 1
        current_rows = int(math.sqrt(current_total))
        current_columns = int(math.ceil(current_total / current_rows))

    max_features = current_total
    print("max_features", max_features)

    top_25_features = df.head(max_features)

    selected_feature_names = top_25_features['var_name'].tolist()

    selected_features = features[selected_feature_names]

    dataset, categorical, numeric = identify_feature_type(selected_features, categorical_threshold=0.05,
                                                          impute_missing=False)

    num_row = int(math.sqrt(max_features))
    num_col = int(math.ceil(max_features / num_row))

    logger.info(
        f"Your dataset {args.dataset} after doing FS has a total of {max_features} features: Images will be generated with a size - ({num_col},{num_row})")

    csv_file_path = str(Path.joinpath(consts.PATH_PROJECT_TAB_TO_IMAGE, f'image_{args.noise_type}_{args.dataset}'))

    best_noisy_data_generator = IGTD_Transformer(num_row=num_row,
                                                 num_col=num_col,
                                                 save_image_size=1,
                                                 max_step=1000,
                                                 val_step=200,
                                                 result_dir=csv_file_path,
                                                 numericas=numeric,
                                                 categoricas=categorical,
                                                 error='abs',
                                                 save_path=csv_file_path,
                                                 type_corr=args.corr,
                                                 )
    noisy_data = best_noisy_data_generator.transform(selected_features)