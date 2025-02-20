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


from utils.NoiseGeneratorTransformer import NoiseDataGenerator
from utils.loader import load_preprocessed_dataset
from utils.preprocessing import identify_binary_and_numeric_features, identify_feature_type
import utils.consts as consts

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def process_iteration_pipeline(args):

    X_original, y_label, numeric_vars, categorical_vars, n_jobs, flag_verbose = args
    noise_data_generator = NoiseDataGenerator(numeric_vars=numeric_vars, categorical_vars=categorical_vars)

    pipeline = Pipeline(steps=[
        ('noisy_data', noise_data_generator),
        ('classifier', LogisticRegression(max_iter=1000, solver='liblinear', penalty='l1', random_state=0))
        # ('classifier', DecisionTreeClassifier(random_state=0))
    ])
    
    param_grid = {
        'noisy_data__sigma': np.arange(0.05, 0.4, 0.05),
        'noisy_data__noise_power': np.arange(0.05, 0.4, 0.05),
        'classifier__C': np.logspace(-1.5, 0.4, 10),
        # 'classifier__max_depth': range(2, 11, 2),
        # 'classifier__min_samples_split': range(2, 40),
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
    parser.add_argument('--n_jobs', default=4, type=int)
    parser.add_argument('--n_vars', default=3, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    return parser.parse_args()


if __name__ == '__main__':

    start_time = time.time()

    parser = argparse.ArgumentParser(description='noise generator experiments')
    args = parse_arguments(parser)

    path_dataset, df_features, y_label = load_preprocessed_dataset(args.dataset)
    dataset, categorical_vars, numeric_vars = identify_feature_type(df_features,
                                                                    categorical_threshold=0.05,
                                                                    impute_missing=False
                                                                    )

    X_original = df_features
    num_vars = args.n_vars
    best_results = []
    datasets = []

    # noise_power = {}  
    # sigma = {}  


    n_procs = cpu_count()
    n_jobs = n_procs if args.n_jobs == -1 or args.n_jobs > n_procs else args.n_jobs
    logger.info('The selected n_jobs: {}'.format(n_jobs))

    pool = Pool(processes=n_jobs)

    for _ in range(num_vars):
        best_params, best_score = process_iteration_pipeline((X_original,
                                                              y_label,
                                                              numeric_vars,
                                                              categorical_vars,
                                                              n_jobs,
                                                              args.verbose))
        best_results.append(best_score)

        best_sigma = best_params['noisy_data__sigma']
        best_noise_power = best_params['noisy_data__noise_power']
        best_C = best_params['classifier__C']

        logger.info(
            "Best hyperparameters: sigma={}, noise-percentage={}, C={}".format(best_sigma, best_noise_power, best_C)
        )

        noisy_data_generator = NoiseDataGenerator(sigma=best_sigma,
                                                  noise_power=best_noise_power,
                                                  numeric_vars=numeric_vars,
                                                  categorical_vars=categorical_vars
                                                  )
        X_original = noisy_data_generator.transform(X_original.copy())
        datasets.append(X_original)

    pool.close()
    pool.join()

    print(best_results)
    max_valor = max(best_results)
    indice_max = best_results.index(max_valor)
    print(f"Número de variables ruidosas generadas por cada variable original que ofrece mejores resultados: {indice_max + 1}")
    noisy_data_best = datasets[indice_max]
    noisy_data_with_labels = pd.concat([noisy_data_best, pd.Series(y_label, name='label')], axis=1)

    csv_file_path = str(Path.joinpath(consts.PATH_PROJECT_NOISY_DATASET, f'noisy_dataset_{args.dataset}.csv'))
    noisy_data_with_labels.to_csv(csv_file_path, index=False)

    elapsed_time = time.time() - start_time
    print("Tiempo de ejecución: {} segundos".format(elapsed_time))
