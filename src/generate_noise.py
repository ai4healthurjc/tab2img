import argparse
from utils.NoiseGeneratorTransformer import NoiseDataGenerator
import utils.consts as consts
from utils.loader import load_preprocessed_dataset, normalize_dataframe
from utils.preprocessing import identify_binary_and_numeric_features
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import LogisticRegression
import coloredlogs
import logging



logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def parse_arguments(parser):
    parser.add_argument('--dataset', default='crx', type=str)
    #parser.add_argument('--sigma', default=0.15, type=float)
    #parser.add_argument('--noise_percentage', default=10, type=int)
    return parser.parse_args()


parser = argparse.ArgumentParser(description='noise generator experiments')
args = parse_arguments(parser)

path_dataset, features, y_label = load_preprocessed_dataset(args.dataset)
binarias, numericas = identify_binary_and_numeric_features(features)
# Carga tus datos reales
X = features  

# Define el transformador personalizado
noisy_data_generator = NoiseDataGenerator(numericas=numericas, binarias=binarias)

# Define el pipeline con el transformador y el clasificador
pipeline = Pipeline(steps=[('noisy_data', noisy_data_generator), ('classifier', LogisticRegression(max_iter=1000, solver='liblinear', penalty='l1',))])

# Define los parámetros a ajustar
param_grid = {
    'noisy_data__sigma': np.arange(0.05, 0.4, 0.05),  # Valores de sigma a probar
    'noisy_data__noise_percentage':  np.arange(0.05, 0.4, 0.05),  # Valores de noise_percentage a probar
    'classifier__C': np.logspace(-1.5, 0.4, 10),  # Valores de C a probar
    }

# Configura la búsqueda de hiperparámetros
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, scoring='roc_auc', verbose=1)

logger.info('Training with Lasso, dataset: {}, params: {}'.format(args.dataset, grid_search.get_params()))

# Realiza la búsqueda de hiperparámetros en tus datos reales
grid_search.fit(X, y_label)


# Obtén los mejores hiperparámetros
best_sigma = grid_search.best_params_['noisy_data__sigma']
best_noise_percentage = grid_search.best_params_['noisy_data__noise_percentage']

print(f"Mejores hiperparámetros encontrados: Sigma = {best_sigma}, Noise Percentage = {best_noise_percentage}")

best_noisy_data_generator = NoiseDataGenerator(sigma=best_sigma, noise_percentage=best_noise_percentage, numericas=numericas, binarias=binarias)
noisy_data = best_noisy_data_generator.transform(X)
noisy_data_with_labels = pd.concat([noisy_data, pd.Series(y_label, name='y_label')], axis=1)

csv_file_path = str(Path.joinpath(consts.PATH_PROJECT_NOISY_DATASET, f'noisy_dataset_{args.dataset}.csv'))
noisy_data_with_labels.to_csv(csv_file_path, index=False)