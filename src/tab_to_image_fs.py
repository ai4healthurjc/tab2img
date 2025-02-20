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
from utils.IGTDTransformer import IGTDTransformer
from utils.preprocessing import identify_feature_type
# import utils.feature_selection_bootstrap as fs
from features.ensemble_fs import perform_ensemble_fs
import utils.consts as consts
from utils.plotter import plot_corr_mixed_matrix
from utils.dissimilarity import compute_corr_mixed_dataset_2
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def parse_arguments(parser):
    parser.add_argument('--dataset', default='ionos', type=str)
    parser.add_argument('--noise_type', default='homogeneous', type=str)
    parser.add_argument('--n_boots', default=50, type=int)
    parser.add_argument('--n_jobs', default=4, type=int)
    parser.add_argument('--max_steps', default=1000, type=int)
    parser.add_argument('--val_step', default=100, type=int)
    parser.add_argument('--fs', default='relief', type=str)
    parser.add_argument('--type_aggregation', default='mean', type=str)
    parser.add_argument('--igtd_err', default='abs', type=str)
    return parser.parse_args()


parser = argparse.ArgumentParser(description='noise generator experiments')
args = parse_arguments(parser)

n_procs = cpu_count()
n_jobs = n_procs if args.n_jobs == -1 or args.n_jobs > n_procs else args.n_jobs
logger.info('n_jobs for tab_to_image_fs: {}'.format(n_jobs))

path_dataset, features, y_label = load_dataset_with_noise(args.dataset, args.noise_type)
dataset, categorical, numeric = identify_feature_type(features, categorical_threshold=0.05, impute_missing=False)

num_features = len(features.columns)  # Número total de características en tu conjunto de datos
print(features.columns)
# Calcula el número de filas y columnas de la imagen basándote en el número de características
# Puedes elegir una estrategia para determinar las dimensiones, por ejemplo, tomar la raíz cuadrada o cualquier otra.
print(num_features)
num_row = int(math.sqrt(num_features))
#num_row=38
num_col = int(math.ceil(num_features / num_row))
#num_col=num_row
print(num_row, num_col)

if num_features == (num_row*num_col):

    logger.info(f"Tu dataset {args.dataset} tiene un total de {num_features} features: Se generarán imágenes de tamaño - ({num_col},{num_row})")
    csv_file_path = str(Path.joinpath(consts.PATH_PROJECT_TAB_TO_IMAGE, f'image_{args.noise_type}_{args.dataset}'))
    # df_corr = compute_corr_mixed_dataset_2(features, categorical, numeric)
    # plot_corr_mixed_matrix(df_corr, flag_save_figure=False, type_linkage='complete')

    best_noisy_data_generator = IGTDTransformer(num_row=num_row,
                                                num_col=num_col,
                                                save_image_size=1,
                                                max_step=1000, val_step=100,
                                                result_dir=csv_file_path,
                                                numericas=numeric,
                                                categoricas=categorical,
                                                error=args.igtd_err
                                                )

    noisy_data = best_noisy_data_generator.transform(features)


        #noisy_data_with_labels = pd.concat([noisy_data, pd.Series(y_label, name='y_label')], axis=1)

        #noisy_data_with_labels.to_csv(csv_file_path, index=False)
else:
    logger.info(f"Con el número de variables de tu dataset con ruido {args.dataset} no se puede generar un tamaño de imagen válido. Se aplicarán técnicas de Feature Selection para generar un tamaño válido.")

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
    current_filas = int(math.sqrt(current_total))
    current_columnas = int(math.ceil(current_total / current_filas))

    while current_filas * current_columnas > current_total:
        current_total -= 1
        current_filas = int(math.sqrt(current_total))
        current_columnas = int(math.ceil(current_total / current_filas))

    max_features = current_total
    print("max_features", max_features)

    top_25_features = df.head(max_features)
    selected_feature_names = top_25_features['var_name'].tolist()
    df_selected_features = features[selected_feature_names]
    dataset, categorical, numeric = identify_feature_type(df_selected_features,
                                                          categorical_threshold=0.05,
                                                          impute_missing=False)

    num_row = int(math.sqrt(max_features))
    num_col = int(math.ceil(max_features / num_row))

    logger.info(f"Tu dataset {args.dataset} tras hacer FS tiene un total de {max_features} features: Se generarán imágenes de tamaño - ({num_col},{num_row})")

    csv_file_path = str(Path.joinpath(consts.PATH_PROJECT_TAB_TO_IMAGE, f'image_{args.noise_type}_{args.dataset}'))

    best_noisy_data_generator = IGTDTransformer(num_row=num_row, num_col=num_col, save_image_size=1,
                                                max_step=args.max_steps, val_step=args.val_step,
                                                result_dir=csv_file_path,
                                                numericas=numeric, categoricas=categorical, error=args.igtd_err)

    noisy_data = best_noisy_data_generator.transform(df_selected_features)



