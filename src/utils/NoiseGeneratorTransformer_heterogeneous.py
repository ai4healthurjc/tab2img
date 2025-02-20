import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import product


class NoiseDataGeneratorH(BaseEstimator, TransformerMixin):

    def __init__(self,
                 sigma=0.15,
                 noise_power=0.1,
                 numeric_vars=None,
                 categorical_vars=None
                 ):

        self.sigma = sigma
        self.noise_power = noise_power
        self.numeric_vars = numeric_vars
        self.categorical_vars = categorical_vars
        self.num_of_variables_to_noise = 1  # Número de variables ruidosas a generar por cada variable original

    def _generate_gaussian_noise(self, X):
        # Aplicar ruido gaussiano a las variables numéricas
        X_numeric = X[self.numeric_vars].copy()
        gaussian_noise = np.random.normal(0, self.sigma, X_numeric.shape)
        noisy_numeric_data = X_numeric + gaussian_noise
        noisy_numeric_data.columns = [f'{col}_gaussian_{self.sigma}' for col in X_numeric.columns]
        return noisy_numeric_data

    def _generate_swap_noise(self, X):
        # Aplicar ruido de intercambio a las variables categóricas
        noisy_categorical_data = X[self.categorical_vars].copy()

        num_samples, num_features = noisy_categorical_data.shape
        num_swap = int(num_features * self.noise_power)

        X_array = X.to_numpy()

        for i in range(num_samples):
            values_to_swap = np.arange(num_features)
            np.random.shuffle(values_to_swap)
            swap_indices = values_to_swap[:num_swap]
            swap_values = X_array[i, swap_indices]

            # Mezclar los valores para el intercambio
            np.random.shuffle(swap_values)
            # Actualizar los valores correspondientes en los datos ruidosos
            noisy_categorical_data.iloc[i, swap_indices] = swap_values

        for col in noisy_categorical_data.columns:
            noisy_categorical_data.rename(columns={col: f"{col}_{self.noise_power}"}, inplace=True)

        return noisy_categorical_data

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Generar ruido gaussiano para las variables numéricas
        noisy_numeric_data = self._generate_gaussian_noise(X)

        # Generar ruido de intercambio para las variables categóricas
        noisy_categorical_data = self._generate_swap_noise(X)

        # Combinar los datos originales con el ruido generado
        combined_data = X.copy()
        combined_data[noisy_numeric_data.columns] = noisy_numeric_data
        combined_data[noisy_categorical_data.columns] = noisy_categorical_data

        return combined_data

    def transform_column(self, column):
        # Generar ruido gaussiano para la columna (variable) dada
        if column.name in self.numeric_vars:
            return column + np.random.normal(0, self.sigma, len(column))
        elif column.name in self.categorical_vars:
            # Aplicar ruido de intercambio solo para las variables categóricas
            values_to_swap = column.copy().to_numpy()
            np.random.shuffle(values_to_swap)
            return values_to_swap
        else:
            return column


