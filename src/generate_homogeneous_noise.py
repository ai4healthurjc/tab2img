import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
import coloredlogs
import logging
import time
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import pairwise_distances
from utils.NoiseGeneratorTransformer_prueba import NoiseDataGenerator
from utils.loader import load_preprocessed_dataset
from utils.preprocessing import identify_feature_type
import utils.consts as consts
from utils.dissimilarity import compute_corr_mixed_dataset_2

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

# def process_iteration_pipeline(args):
#     X_original, y_label, numeric_vars, categorical_vars, n_jobs, flag_verbose, max_noise_generations = args
#     all_best_params = {}

#     for i, col in enumerate(X_original):
#         print(col)
#         best_params_list = all_best_params.get(col, [])

#         for _ in range(int(max_noise_generations)):
#             if col in numeric_vars:
#                 noise_data_generator = NoiseDataGenerator(numeric_vars=[col], categorical_vars=[])

#                 # Excludes the best previous values of 'noisy_data__sigma'
#                 excluded_values = [params.get('noisy_data__sigma', None) for params in best_params_list]
#                 excluded_values = [value for value in excluded_values if value is not None]
#                 param_grid = {
#                     'noisy_data__sigma': np.setdiff1d(np.arange(0.05, 0.4, 0.05), excluded_values),
#                     'classifier__C': np.logspace(-1.5, 0.4, 10),
#                 }
#             elif col in categorical_vars:
#                 noise_data_generator = NoiseDataGenerator(numeric_vars=[], categorical_vars=[col])

#                 # Excludes the best previous values of 'noisy_data__noise_power'
#                 excluded_values = [params.get('noisy_data__noise_power', None) for params in best_params_list]
#                 excluded_values = [value for value in excluded_values if value is not None]
#                 param_grid = {
#                     'noisy_data__noise_power': np.setdiff1d(np.arange(0.05, 0.4, 0.05), excluded_values),
#                     'classifier__C': np.logspace(-1.5, 0.4, 10),
#                 }
#             else:
#                 continue
#             print(param_grid)
#             pipeline = Pipeline(steps=[
#                 ('noisy_data', noise_data_generator),
#                 ('classifier', LogisticRegression(max_iter=1000, solver='liblinear', penalty='l1', random_state=0))
#             ])

#             grid_search = GridSearchCV(estimator=pipeline,
#                                        param_grid=param_grid,
#                                        cv=3,
#                                        scoring='roc_auc',
#                                        verbose=flag_verbose,
#                                        n_jobs=n_jobs)

#             grid_search.fit(X_original, y_label)
#             best_params_list.append(grid_search.best_params_)

#         # Updates the best parameters for next iteration
#         all_best_params[col] = best_params_list
#         print(all_best_params)


#     return all_best_params


# def parse_arguments(parser):
#     parser.add_argument('--dataset', default='crx', type=str)
#     parser.add_argument('--n_jobs', default=4, type=int)
#     parser.add_argument('--n_vars', default=3, type=int)
#     parser.add_argument('--verbose', default=1, type=int)
#     parser.add_argument('--fs', default='relief', type=str)
#     parser.add_argument('--n_boots', default=100, type=int)
#     parser.add_argument('--agg_func', default='mean', type=str)

#     return parser.parse_args()


# if __name__ == '__main__':

#     start_time = time.time()

#     parser = argparse.ArgumentParser(description='noise generator experiments')
#     args = parse_arguments(parser)

#     path_dataset, df_features, y_label = load_preprocessed_dataset(args.dataset)
#     dataset, categorical_vars, numeric_vars = identify_feature_type(df_features,
#                                                                     categorical_threshold=0.05,
#                                                                     impute_missing=False
#                                                    )

#     num_vars = args.n_vars

#     X_original = df_features
#     n_procs = cpu_count()
#     n_jobs = n_procs if args.n_jobs == -1 or args.n_jobs > n_procs else args.n_jobs

    
#     logger.info('The selected n_jobs: {}'.format(n_jobs))

#     best_params = process_iteration_pipeline((X_original, y_label, numeric_vars, categorical_vars,
#                                               n_jobs, args.verbose, num_vars))
#     print("Best parameters for each variable:")
#     print(best_params)

    
#     transformed_datasets = []
#     for col, best_params_list in best_params.items():
#         for idx, best_params in enumerate(best_params_list):
#             # Extracts the relevant parameters
#             sigma = best_params.get('noisy_data__sigma', None)
#             noise_power = best_params.get('noisy_data__noise_power', None)

#             # If there is a value of 'sigma', generates noise with 'sigma' and displays the result
#             if sigma is not None:
#                 noise_data_generator = NoiseDataGenerator(numeric_vars=[col], categorical_vars=[], sigma=sigma)
#                 X_noisy = noise_data_generator.fit_transform(X_original, y_label)
#                 print(f"Generated noise for {col} - Feature {idx + 1}, Sigma: {sigma}")

#             # If there is a value of 'noise_power', generate noise with 'noise_power' and display the result
#             elif noise_power is not None:
#                 noise_data_generator = NoiseDataGenerator(numeric_vars=[], categorical_vars=[col],
#                                                           noise_power=noise_power)
#                 X_noisy = noise_data_generator.fit_transform(X_original)
#                 print(f"Generated noise for {col} - Feature {idx + 1}, Noise Power: {noise_power}")

#             transformed_datasets.append(X_noisy)
#             print(transformed_datasets)

#     # Concatenate all transformed datasets
#     final_dataset = pd.concat(transformed_datasets, axis=1)
#     # Delete duplicate columns (based on column name)
#     final_dataset = final_dataset.loc[:, ~final_dataset.columns.duplicated()]
#     print(final_dataset)
#     csv_file_path = str(Path.joinpath(consts.PATH_PROJECT_NOISY_DATASET,
#                                       f"noisy_dataset_homogeneous_{args.dataset}.csv"))
#     final_dataset.to_csv(csv_file_path, index=False)


def process_iteration_pipeline(args):
    df_data, y_label, numeric_vars, categorical_vars, n_jobs, flag_verbose = args
    best_params = {}

    for col in numeric_vars:

        noise_data_generator = NoiseDataGenerator(numeric_vars=[col], categorical_vars=[])

        pipeline = Pipeline(steps=[
            ('noisy_data', noise_data_generator),
            ('classifier', LogisticRegression(max_iter=1000, solver='liblinear', penalty='l1', random_state=0))
        ])
        
        param_grid = {
            'noisy_data__sigma': np.arange(0.05, 0.4, 0.05),
            'classifier__C': np.logspace(-1.5, 0.4, 10),
        }

        if len(np.unique(y_label)) == 2:
            score = 'roc_auc'
        else:
            score = 'accuracy'

        print(score)

        grid_search = GridSearchCV(estimator=pipeline,
                                   param_grid=param_grid,
                                   cv=5,
                                   scoring=score,
                                   verbose=flag_verbose,
                                   n_jobs=n_jobs
                                   )

        grid_search.fit(df_data, y_label)
        best_params[col] = grid_search.best_params_

    for col in categorical_vars:
        noise_data_generator = NoiseDataGenerator(numeric_vars=[], categorical_vars=[col])
        pipeline = Pipeline(steps=[
            ('noisy_data', noise_data_generator),
            ('classifier', LogisticRegression(max_iter=1000, solver='liblinear', penalty='l1', random_state=0))
        ])
        
        param_grid = {
            'noisy_data__noise_power': np.arange(0.05, 0.1, 0.02),
            'classifier__C': np.logspace(-1.5, 0.4, 10),
        }

        if len(np.unique(y_label)) == 2:
            score = 'roc_auc'
        else:
            score = 'accuracy'

        grid_search = GridSearchCV(estimator=pipeline,
                                   param_grid=param_grid,
                                   cv=5,
                                   scoring=score,
                                   verbose=flag_verbose,
                                   n_jobs=n_jobs)

        grid_search.fit(df_data, y_label)
        print(grid_search.cv_results_)

        best_params[col] = grid_search.best_params_

    return best_params


def parse_arguments(parser):
    parser.add_argument('--dataset', default='crx', type=str)
    parser.add_argument('--categorical_encoding', default='helmert', type=str)
    parser.add_argument('--n_jobs', default=4, type=int)
    parser.add_argument('--n_vars', default=3, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    return parser.parse_args()


if __name__ == '__main__':

    start_time = time.time()

    cmd_parser = argparse.ArgumentParser(description='noise generator experiments')
    args = parse_arguments(cmd_parser)

    path_dataset, df_features, y_label = load_preprocessed_dataset(args.dataset, args.categorical_encoding)
    dataset, categorical_vars, numeric_vars = identify_feature_type(df_features,
                                                                    categorical_threshold=0.05,
                                                                    impute_missing=False)
    num_vars = args.n_vars
    best_noise = []
    df_original = df_features
    n_procs = cpu_count()
    n_jobs = n_procs if args.n_jobs == -1 or args.n_jobs > n_procs else args.n_jobs
    logger.info('The selected n_jobs: {}'.format(n_jobs))

    pool = Pool(processes=n_jobs)
    
    for _ in range(num_vars):
        best_params = process_iteration_pipeline((df_original, y_label, numeric_vars,
                                                  categorical_vars, n_jobs, args.verbose))
        print("Best parameters for each variable:")
        print(best_params)
        best_noise.append(best_params)

    pool.close()
    pool.join()

    transformed_datasets = []
    datasets = []

    pool = Pool(processes=n_jobs)
    count = -1
    for i in best_noise:
        count += 1
        for variable, params in i.items():

            sigma = params.get('noisy_data__sigma')
            noise_power = params.get('noisy_data__noise_power')
            numeric_vars_ = [variable] if variable in numeric_vars else []
            categorical_vars_ = [variable] if variable in categorical_vars else []

            noise_data_generator = NoiseDataGenerator(
                numeric_vars=numeric_vars_,
                categorical_vars=categorical_vars_,
                sigma=sigma,
                noise_power=noise_power,
                generation_count=count
            )

            df_original = noise_data_generator.transform(df_original)
            transformed_datasets.append(df_original.copy())

        df_original = pd.concat(transformed_datasets, axis=1)
        df_original = df_original.loc[:, ~df_original.columns.duplicated()]
        print(df_original)
        datasets.append(df_original.copy())

    pool.close()
    pool.join()

    best_n_noisy = []
    for noisy_features in datasets:
        param_grid = {
            'C': np.logspace(-1.5, 0.4, 10),
        }

        if len(np.unique(y_label)) == 2:
            score = 'roc_auc'
        else:
            score = 'accuracy'

        grid_search = GridSearchCV(estimator=LogisticRegression(max_iter=1000, solver='liblinear',
                                                                penalty='l1', random_state=0),
                                   param_grid=param_grid,
                                   cv=5,
                                   scoring=score,
                                   verbose=1,
                                   n_jobs=n_jobs)

        grid_search.fit(noisy_features, y_label)
        best_n_noisy.append(grid_search.best_score_)
    
    max_valor = max(best_n_noisy)
    index_max = best_n_noisy.index(max_valor)
    print(f"Number of noisy variables generated for each original variable giving better results: {index_max + 1}")
    noisy_data_best = datasets[index_max]
    noisy_data_with_labels = pd.concat([noisy_data_best, pd.Series(y_label, name='label')], axis=1)

    csv_file_path = str(Path.joinpath(consts.PATH_PROJECT_NOISY_DATASET,
                                      f'noisy_dataset_homogeneous_{args.dataset}.csv'))
    noisy_data_with_labels.to_csv(csv_file_path, index=False)

    elapsed_time = time.time() - start_time
    print("Execution time: {} seconds".format(elapsed_time))

#     # dataset, categorical, numeric = identify_feature_type(df_features, categorical_threshold=0.05, impute_missing=False)
#     # original_corr_matrix = compute_corr_mixed_dataset_2(df_features, categorical, numeric)

#     # # Initialize best distance as infinity to compare with calculated distances
#     # best_distance = float('inf')
#     # best_noisy_dataset = None
#     # counter=0
#     # for noisy_features in datasets:
#     #     counter+=1
#     #     dataset_noisy, categorical_noisy, numeric_noisy = identify_feature_type(noisy_features, categorical_threshold=0.05, impute_missing=False)
#     #     noisy_corr_matrix = compute_corr_mixed_dataset_2(noisy_features, categorical_noisy, numeric_noisy)

#     #     original_corr_matrix_aligned = original_corr_matrix.loc[noisy_corr_matrix.index, noisy_corr_matrix.columns]
#     #     # Calculate the correlation matrix of the noisy data set
        
#     #     # Calculate the Euclidean distance between correlation matrices
#     #     distance = pairwise_distances([original_corr_matrix_aligned, noisy_corr_matrix], metric='correlation')

#     #     # Update the best noisy dataset if we find a smaller distance
#     #     if distance < best_distance:
#     #         best_distance = distance
#     #         print(counter)
#     #         best_noisy_dataset = noisy_features

#     # print(best_noisy_dataset)
