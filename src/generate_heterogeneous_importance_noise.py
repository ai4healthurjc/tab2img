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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler

import features.ensemble_fs as fs
from utils.NoiseGeneratorTransformer_prueba import NoiseDataGenerator
from utils.loader import load_preprocessed_dataset
from utils.preprocessing import identify_feature_type
import utils.consts as consts

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def process_iteration_pipeline(args):
    X_fs, X_original, y_label, numeric_vars, categorical_vars, n_jobs, flag_verbose, max_noise_generations = args
    # best_params = {}

    feature_importances = X_fs['score']
    scaler = MinMaxScaler()
    normalized_importances = scaler.fit_transform(feature_importances.values.reshape(-1, 1)).flatten()
    print(normalized_importances)
    # Iter on selected characteristics
    # current_best_params = []
    all_best_params = {}

    pool = Pool(processes=n_jobs)
    for i, col in enumerate(X_fs['var_name']):
        best_params_list = all_best_params.get(col, [])
        # best_params = {}
        # print(i)
        print(col)
        # Decide how many times to generate noise for each feature according to its importance
        num_noise_generations = np.round(normalized_importances[i] * max_noise_generations)
        print(num_noise_generations)

        # generation_count=0
        # Generates and adds noise to the feature according to importance
        for _ in range(int(num_noise_generations)):
            # print(_)
            if col in numeric_vars:
                noise_data_generator = NoiseDataGenerator(numeric_vars=[col], categorical_vars=[])
                param_grid = {
                    'noisy_data__sigma': np.arange(0.05, 0.4, 0.05),
                    'classifier__C': np.logspace(-1.5, 0.4, 10),
                }
            elif col in categorical_vars:
                noise_data_generator = NoiseDataGenerator(numeric_vars=[], categorical_vars=[col])
                param_grid = {
                    'noisy_data__noise_power': np.arange(0.05, 0.1, 0.02),
                    'classifier__C': np.logspace(-1.5, 0.4, 10),
                }
            else:
                continue  # Handling of the case where cabbage is not on any of the lists

            pipeline = Pipeline(steps=[
                ('noisy_data', noise_data_generator),
                ('classifier', LogisticRegression(max_iter=1000, solver='liblinear', penalty='l1', random_state=_))
            ])

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
                                       n_jobs=n_jobs)

            grid_search.fit(X_original, y_label)
            # best_params[col] = grid_search.best_params_
            # print(best_params)
            best_params_list.append(grid_search.best_params_)

#         # Updates the best parameters for next iteration
#         all_best_params[col] = best_params_list
#         print(all_best_params)

            all_best_params[col] = best_params_list
            print(all_best_params)
    pool.close()
    pool.join()

    # best_params[col] = grid_search.best_params_

    return all_best_params


def parse_arguments(parser):
    parser.add_argument('--dataset', default='crx', type=str)
    parser.add_argument('--categorical_encoding', default='count', type=str)
    parser.add_argument('--n_jobs', default=4, type=int)
    parser.add_argument('--n_vars', default=3, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--fs', default='relief', type=str)
    parser.add_argument('--n_boots', default=100, type=int)
    parser.add_argument('--agg_func', default='mean', type=str)

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

    X_original = df_features
    n_procs = cpu_count()
    n_jobs = n_procs if args.n_jobs == -1 or args.n_jobs > n_procs else args.n_jobs

    rus = RandomUnderSampler(sampling_strategy='all', random_state=42)
    X_res, y_res = rus.fit_resample(df_features, y_label)
    print('Resampled dataset shape %s' % y_res.value_counts())
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(X_res)
    
    y_res = np.array(y_res)
    v_feature_names = df_features.columns
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

    logger.info('The selected n_jobs: {}'.format(n_jobs))

    best_params = process_iteration_pipeline((df, X_original, y_label, numeric_vars, categorical_vars,
                                              n_jobs, args.verbose, num_vars))
    print("Best parameters for each variable:")
    print(best_params)

    pool = Pool(processes=n_jobs)
    transformed_datasets = []
    # generation_count=0
    for col, best_params_list in best_params.items():
        for idx, best_params in enumerate(best_params_list):
            # generation_count+=1
            # Extracts the relevant parameters
            sigma = best_params.get('noisy_data__sigma', None)
            noise_power = best_params.get('noisy_data__noise_power', None)

            # If there is a value of 'sigma', generates noise with 'sigma' and displays the result
            if sigma is not None:
                noise_data_generator = NoiseDataGenerator(numeric_vars=[col], categorical_vars=[],
                                                          generation_count=idx, sigma=sigma)
                X_noisy = noise_data_generator.fit_transform(X_original, y_label)
                print(f"Generated noise for {col} - Feature {idx + 1}, Sigma: {sigma}")

            # If there is a value of 'noise_power', generate noise with 'noise_power' and display the result
            elif noise_power is not None:
                noise_data_generator = NoiseDataGenerator(numeric_vars=[], categorical_vars=[col],
                                                          generation_count=idx, noise_power=noise_power)
                X_noisy = noise_data_generator.fit_transform(X_original)
                print(f"Generated noise for {col} - Feature {idx + 1}, Noise Power: {noise_power}")

            transformed_datasets.append(X_noisy)
    pool.close()
    pool.join()

    # Concatenate all transformed datasets
    final_dataset = pd.concat(transformed_datasets, axis=1)
    # Delete duplicate columns (based on column name)
    final_dataset = final_dataset.loc[:, ~final_dataset.columns.duplicated()]
    print(final_dataset)
    final_dataset = pd.concat([final_dataset, pd.Series(y_label, name='label')], axis=1)

    csv_file_path = str(Path.joinpath(consts.PATH_PROJECT_NOISY_DATASET,
                                      f'noisy_dataset_heterogeneous_{args.dataset}.csv'))
    final_dataset.to_csv(csv_file_path, index=False)

    elapsed_time = time.time() - start_time
    print("Time of execution: {} seconds".format(elapsed_time))


# DO NOT REMOVE

# def process_iteration_pipeline(args):
#     X_fs, X_original, y_label, numeric_vars, categorical_vars, n_jobs, flag_verbose, max_noise_generations = args
#     all_best_params = {}

#     feature_importances = X_fs['score']
#     scaler = MinMaxScaler()
#     normalized_importances = scaler.fit_transform(feature_importances.values.reshape(-1, 1)).flatten()
#     print(normalized_importances)

#     # Iter on selected characteristics
#     for i, col in enumerate(X_fs['var_name']):
#         print(col)
#         best_params_list = all_best_params.get(col, [])

#         # Decide how many times to generate noise for each feature according to its importance
#         num_noise_generations = np.round(normalized_importances[i] * max_noise_generations)
#         print(num_noise_generations)

#         for _ in range(int(num_noise_generations)):
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
#                     'noisy_data__noise_power': np.setdiff1d(np.arange(0.05, 0.25, 0.05), excluded_values),
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
