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
import numpy as np
from sklearn.utils import shuffle
import random

def process_iteration_pipeline(args):
    X_original, y_label, numeric_vars, categorical_vars, n_jobs, flag_verbose = args
    best_params = {}

    # Definir las probabilidades para recibir ruido para cada variable
    prob_ruido_numeric = {var: random.uniform(0, 1) for var in numeric_vars}
    prob_ruido_categorical = {var: random.uniform(0, 1) for var in categorical_vars}

    for col in numeric_vars:
        # Verificar si la variable recibe ruido según la probabilidad
        if random.uniform(0, 1) <= prob_ruido_numeric[col]:
            print(col)
            noise_data_generator = NoiseDataGenerator(numeric_vars=[col], categorical_vars=[])
            pipeline = Pipeline(steps=[
                ('noisy_data', noise_data_generator),
                ('classifier', LogisticRegression(max_iter=1000, solver='liblinear', penalty='l1', random_state=0))
            ])

            param_grid = {
                'noisy_data__sigma': np.arange(0.05, 0.4, 0.05),
                'classifier__C': np.logspace(-1.5, 0.4, 10),
            }

            grid_search = GridSearchCV(estimator=pipeline,
                                       param_grid=param_grid,
                                       cv=3,
                                       scoring='roc_auc',
                                       verbose=flag_verbose,
                                       n_jobs=n_jobs)

            grid_search.fit(X_original, y_label)
            best_params[col] = grid_search.best_params_

    for col in categorical_vars:
        # Verificar si la variable recibe ruido según la probabilidad
        if random.uniform(0, 1) <= prob_ruido_categorical[col]:
            print(col)
            noise_data_generator = NoiseDataGenerator(numeric_vars=[], categorical_vars=[col])
            pipeline = Pipeline(steps=[
                ('noisy_data', noise_data_generator),
                ('classifier', LogisticRegression(max_iter=1000, solver='liblinear', penalty='l1', random_state=0))
            ])

            param_grid = {
                'noisy_data__noise_power': np.arange(0.05, 0.4, 0.05),
                'classifier__C': np.logspace(-1.5, 0.4, 10),
            }

            grid_search = GridSearchCV(estimator=pipeline,
                                       param_grid=param_grid,
                                       cv=3,
                                       scoring='roc_auc',
                                       verbose=flag_verbose,
                                       n_jobs=n_jobs)

            grid_search.fit(X_original, y_label)
            best_params[col] = grid_search.best_params_

    return best_params


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
                                                                    categorical_threshold=0.01,
                                                                    impute_missing=False
                                                   )
    num_vars = args.n_vars
    best_noise=[]
    X_original = df_features
    n_procs = cpu_count()
    n_jobs = n_procs if args.n_jobs == -1 or args.n_jobs > n_procs else args.n_jobs
    
    pool = Pool(processes=n_jobs)
    
    for _ in range(num_vars):
    
        logger.info('The selected n_jobs: {}'.format(n_jobs))

        best_params = process_iteration_pipeline((X_original, y_label, numeric_vars, categorical_vars, n_jobs, args.verbose))
        print("Best parameters for each variable:")
        print(best_params)
        best_noise.append(best_params)
        print(best_noise)

    pool.close()
    pool.join()

    transformed_datasets = []
    datasets=[]

    pool = Pool(processes=n_jobs)

    for i in best_noise:
        for variable, params in i.items():

            sigma = params.get('noisy_data__sigma')
            noise_power = params.get('noisy_data__noise_power')

            # Validar si la variable es numérica o categórica
            numeric_vars_ = [variable] if variable in numeric_vars else []
            categorical_vars_ = [variable] if variable in categorical_vars else []

            # Inicializar el NoiseDataGenerator solo con los parámetros específicos de ruido
            noise_data_generator = NoiseDataGenerator(
                numeric_vars=numeric_vars_,
                categorical_vars=categorical_vars_,
                sigma=sigma,
                noise_power=noise_power
            )

            X_original = noise_data_generator.transform(X_original)
            transformed_datasets.append(X_original.copy())  

        X_original = pd.concat(transformed_datasets, axis=1)
        X_original = X_original.loc[:, ~X_original.columns.duplicated()]
        print(X_original)
        datasets.append(X_original.copy())
        print(datasets)
    pool.close()
    pool.join()

    pool = Pool(processes=n_jobs)

    best_n_noisy=[]
    for noisy_features in datasets:
        param_grid = {
            'C': np.logspace(-1.5, 0.4, 10),
        }

        grid_search = GridSearchCV(estimator=LogisticRegression(max_iter=1000, solver='liblinear', penalty='l1', random_state=0),
                                   param_grid=param_grid,
                                   cv=3,
                                   scoring='roc_auc',
                                   verbose=1,
                                   n_jobs=n_jobs)
        scaler = StandardScaler()
    
        X_train = scaler.fit_transform(noisy_features)

        grid_search.fit(X_train, y_label)
        best_n_noisy.append(grid_search.best_score_)
    pool.close()
    pool.join()
    
    max_valor = max(best_n_noisy)
    indice_max = best_n_noisy.index(max_valor)
    print(f"Número de variables ruidosas generadas por cada variable original que ofrece mejores resultados: {indice_max + 1}")
    noisy_data_best = datasets[indice_max]
    noisy_data_with_labels = pd.concat([noisy_data_best, pd.Series(y_label, name='label')], axis=1)

    csv_file_path = str(Path.joinpath(consts.PATH_PROJECT_NOISY_DATASET, f'noisy_dataset_random_heterogeneous_{args.dataset}.csv'))
    noisy_data_with_labels.to_csv(csv_file_path, index=False)

    elapsed_time = time.time() - start_time
    print("Tiempo de ejecución: {} segundos".format(elapsed_time))




