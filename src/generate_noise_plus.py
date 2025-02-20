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
import time 


logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def parse_arguments(parser):
    parser.add_argument('--dataset', default='crx', type=str)
    return parser.parse_args()


parser = argparse.ArgumentParser(description='noise generator experiments')
args = parse_arguments(parser)

start_time = time.time()  # Registra el tiempo de inicio

path_dataset, features, y_label = load_preprocessed_dataset(args.dataset)
binarias, numericas = identify_binary_and_numeric_features(features)

# Define el número máximo de variables de ruido
num_vars = 3

# Inicializa tus datos reales
X_original = features


best_score = -1  # Inicializa el mejor puntaje
best_hiperparametros = {}
best_noisy_data = None
best_results=[]
datasets=[]

for _ in range(num_vars):  # Ejecutar tres veces
    # Define el transformador personalizado
    noisy_data_generator = NoiseDataGenerator(numericas=numericas, binarias=binarias)

    # Define el pipeline con el transformador y el clasificador
    pipeline = Pipeline(steps=[('noisy_data', noisy_data_generator),
                               ('classifier', LogisticRegression(max_iter=1000, solver='liblinear', penalty='l1',random_state=0))])

    # Define los parámetros a ajustar
    param_grid = {
        'noisy_data__sigma': np.arange(0.05, 0.4, 0.05),  # Valores de sigma a probar
        'noisy_data__noise_percentage':  np.arange(0.05, 0.4, 0.05),  # Valores de noise_percentage a probar
        'classifier__C': np.logspace(-1.5, 0.4, 10),  # Valores de C a probar
    }

    # Configura la búsqueda de hiperparámetros
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, scoring='roc_auc', verbose=1)

    # Realiza la búsqueda de hiperparámetros en tus datos originales
    grid_search.fit(X_original, y_label)

    # Obtén los mejores hiperparámetros
    best_sigma = grid_search.best_params_['noisy_data__sigma']
    best_noise_percentage = grid_search.best_params_['noisy_data__noise_percentage']
    best_C = grid_search.best_params_['classifier__C']
    
    logger.info(f"Mejores hiperparámetros encontrados: Sigma = {best_sigma}, Noise Percentage = {best_noise_percentage}, C = {best_C}")
    # print(f"Mejores hiperparámetros encontrados: Sigma = {best_sigma}, Noise Percentage = {best_noise_percentage}, C = {best_C}")

    best_results.append(grid_search.best_score_)
    

    # Genera un conjunto de datos ruidosos con los mejores hiperparámetros
    noisy_data_generator = NoiseDataGenerator(sigma=best_sigma,
                                              noise_percentage=best_noise_percentage,
                                              numericas=numericas,
                                              binarias=binarias
                                              )
    X_original = noisy_data_generator.transform(X_original.copy())
    print(X_original)
    datasets.append(X_original)


print(best_results)
    # Encuentra el valor máximo en la lista
max_valor = max(best_results)

# Encuentra el índice del valor máximo
indice_max = best_results.index(max_valor)

print(f"Número de variables ruidosas generadas por cada variable original que ofrece mejores resultados: {indice_max + 1}")
noisy_data_best = datasets[indice_max]

noisy_data_with_labels = pd.concat([noisy_data_best, pd.Series(y_label, name='y_label')], axis=1)

csv_file_path = str(Path.joinpath(consts.PATH_PROJECT_NOISY_DATASET, f'noisy_dataset_{args.dataset}.csv'))
noisy_data_with_labels.to_csv(csv_file_path, index=False)

end_time = time.time()  # Registra el tiempo de finalización
elapsed_time = end_time - start_time  # Calcula el tiempo transcurrido
print(f"Tiempo de ejecución: {elapsed_time} segundos")