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
from itertools import product

from utils.NoiseGeneratorTransformer_heterogeneous import NoiseDataGeneratorH
from utils.loader import load_preprocessed_dataset
from utils.preprocessing import identify_feature_type
import utils.consts as consts

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def process_iteration_pipeline(args):
    X_original, y_label, numeric_vars, categorical_vars, n_jobs, flag_verbose = args
    noise_data_generator = NoiseDataGeneratorH(numeric_vars=numeric_vars, categorical_vars=categorical_vars)

    pipeline = Pipeline(steps=[
        ('noisy_data', noise_data_generator),
        ('classifier', LogisticRegression(max_iter=1000, solver='liblinear', penalty='l1', random_state=0))
    ])

    param_grid = {
        'noisy_data__sigma': np.arange(0.05, 0.4, 0.05),
        'noisy_data__noise_power': np.arange(0.05, 0.4, 0.05),
        'classifier__C': np.logspace(-1.5, 0.4, 10),
    }

    grid_search = GridSearchCV(estimator=pipeline,
                               param_grid=param_grid,
                               cv=3,
                               scoring='roc_auc',
                               verbose=flag_verbose,
                               n_jobs=n_jobs
                               )

    grid_search.fit(X_original, y_label)

    return grid_search.best_params_, grid_search.best_score_


def parse_arguments(parser):
    parser.add_argument('--dataset', default='crx', type=str)
    parser.add_argument('--categorical_encoding', default='count', type=str)
    parser.add_argument('--n_jobs', default=4, type=int)
    parser.add_argument('--n_vars', default=3, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='noise generator experiments')
    args = parse_arguments(parser)

    path_dataset, df_features, y_label = load_preprocessed_dataset(args.dataset, args.categorical_encoding)
    dataset, categorical_vars, numeric_vars = identify_feature_type(df_features,
                                                                    categorical_threshold=0.05,
                                                                    impute_missing=False
                                                                    )

    X_original = df_features
    num_vars = args.n_vars
    best_results = []
    datasets = []

    n_procs = cpu_count()
    print(n_procs)
    n_jobs = n_procs if args.n_jobs == -1 or args.n_jobs > n_procs else args.n_jobs
    logger.info('The selected n_jobs: {}'.format(n_jobs))

    pool = Pool(processes=n_jobs)

    # Lista de posibles números de variables ruidosas para cada variable original
    possible_num_vars = range(0, num_vars + 1)

    for combination in product(possible_num_vars, repeat=len(df_features.columns)):
        best_params, best_score = process_iteration_pipeline((X_original,
                                                              y_label,
                                                              numeric_vars,
                                                              categorical_vars,
                                                              n_jobs,
                                                              args.verbose))
        best_results.append((combination, best_score))
        best_sigma = best_params['noisy_data__sigma']
        best_noise_power = best_params['noisy_data__noise_power']

        noisy_data_generator = NoiseDataGeneratorH(sigma=best_sigma,
                                                  noise_power=best_noise_power,
                                                  numeric_vars=numeric_vars,
                                                  categorical_vars=categorical_vars
                                                  )
        X_original = noisy_data_generator.transform(X_original.copy())
        datasets.append(X_original)

    pool.close()
    pool.join()

    print(best_results)
    max_result = max(best_results, key=lambda x: x[1])
    best_combination = max_result[0]

    print(f"Best combination of numbers of noisy variables generated: {best_combination}")

    # Aplica la generación de ruido con cantidades específicas para cada variable original
    noisy_data_generator = NoiseDataGeneratorH(sigma=best_sigma, noise_power=best_noise_power, numeric_vars=numeric_vars, categorical_vars=categorical_vars)
    X_original = df_features.copy()
    
    for num_vars, feature_name in zip(best_combination, X_original.columns):
        if num_vars > 0:
            noisy_data_generator.num_of_variables_to_noise = num_vars
            X_original[feature_name] = noisy_data_generator.transform_column(X_original[feature_name])

    noisy_data_with_labels = pd.concat([X_original, pd.Series(y_label, name='label')], axis=1)

    csv_file_path = str(Path.joinpath(consts.PATH_PROJECT_NOISY_DATASET, f'noisy_dataset_heterogeneous{args.dataset}.csv'))
    noisy_data_with_labels.to_csv(csv_file_path, index=False)

    elapsed_time = time.time() - start_time
    print(f"Execution time: {elapsed_time} seconds")
