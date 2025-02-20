import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NoiseDataGenerator(BaseEstimator, TransformerMixin):

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

    def _create_n_noisy_vars(self, df, noise_power, n_noisy_vars=1):
        range_n_noisy_vars = range(n_noisy_vars)

        df.assign(**{
            f'{col}_({noisy_var})': df[col].apply(self._generate_gaussian_noise, axis=1, args=noise_power)
            for noisy_var in range_n_noisy_vars
            for col in df
        })

    def _generate_gaussian_noise(self, X):

        # Apply Gaussian noise to numeric variables
        X_numeric = X[self.numeric_vars].copy()
        gaussian_noise = np.random.normal(0, self.sigma, X_numeric.shape)
        noisy_numeric_data = X_numeric + gaussian_noise
        noisy_numeric_data.columns = ['{}_gaussian_{}'.format(col, self.sigma) for col in X_numeric.columns]
        
        return noisy_numeric_data

    def _generate_swap_noise(self, X):

            # # Apply noise only for specific categorical variables
            noisy_categorical_data = X[self.categorical_vars].copy()

            num_samples, num_features = noisy_categorical_data.shape
            num_swap = int(num_features * self.noise_power)

            X_array = X.to_numpy()

            for i in range(num_samples):
                values_to_swap = np.arange(num_features)
                np.random.shuffle(values_to_swap)
                swap_indices = values_to_swap[:num_swap]
                swap_values = X_array[i, swap_indices]

                # Shuffle the values for swapping
                np.random.shuffle(swap_values)
                # Update the corresponding values in the noisy data
                # noisy_categorical_data[i, swap_indices] = swap_values
                noisy_categorical_data.iloc[i, swap_indices] = swap_values

            for col in noisy_categorical_data.columns:
                noisy_categorical_data.rename(columns={col: f"{col}_{self.noise_power}"}, inplace=True)

    # def _generate_salt_pepper_noise(self, X):
            
    #         noisy_binary_data = X[self.categorical_vars].copy()
    #         for column in self.categorical_vars:
    #             noisy_binary_data[column] = np.random.choice([0, 1], size=len(X), p=[1 - self.noise_power, self.noise_power])
    #             noisy_binary_data.rename(columns={column: f"{column}_noisy_{self.noise_power}"}, inplace=True)


            # # Aplicar swapping noise a las columnas categóricas manteniendo el número original de categorías
            #     for col in self.categoricas:
            #         unique_values = noisy_categorical_data[col].unique()
            #         total_unique = len(unique_values)

            # # Calcula el número de intercambios basado en un porcentaje de valores únicos
            #         swap_count = max(int(total_unique * self.swap_percentage), 1)  # Al menos 1 swap para evitar swap_count = 0
            #         # unique_values = noisy_categorical_data[col].unique()
            #         # swap_count = int(len(unique_values) * self.swap_percentage)
            #         print(swap_count)

            #         np.random.shuffle(unique_values)
            #         mapping = {original: noisy for original, noisy in zip(unique_values, unique_values[:swap_count])}
            #         print(mapping)
            # noisy_categorical_data[f"{col}_noisy_{self.swap_percentage}"] = noisy_categorical_data[col].map(mapping)

            return noisy_categorical_data

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        noisy_numeric_data = self._generate_gaussian_noise(X)
        noisy_categorical_data = self._generate_swap_noise(X)

        df_combined_data = X.copy()
        # df_combined_data[noisy_numeric_data.columns] = noisy_numeric_data
        # df_combined_data[noisy_categorical_data.columns] = noisy_categorical_data
        df_combined_data = pd.concat((df_combined_data, noisy_numeric_data, noisy_categorical_data), axis=1)

        print(df_combined_data.columns)
        print(df_combined_data.shape)

        # mapping = {original: noisy for original, noisy in zip(unique_values[:swap_count], unique_values)}
        # noisy_categorical_data[col] = noisy_categorical_data[col].map(mapping)
        # print(noisy_categorical_data)
        # noisy_categorical_data[f"{col}_noisy_{self.swap_percentage}"] = noisy_categorical_data[col].map(mapping)

        # noisy_categorical_data = X[self.categoricas].copy()

        # for col in self.categoricas:
        #     for _ in range(self.num_swaps):
        #         unique_values = noisy_categorical_data[col].unique()
        #         np.random.shuffle(unique_values)
        #         mapping = {original: noisy for original, noisy in zip(noisy_categorical_data[col].unique(), unique_values)}
        #         noisy_categorical_data[col] = noisy_categorical_data[col].map(mapping)
        #         noisy_categorical_data.rename(columns={col: f"{col}_{noisy_categorical_data}_noisy_{self.num_swaps}"}, inplace=True)



            # noisy_binary_data[column] = np.random.choice([0, 1], size=len(X), p=[1 - self.noise_percentage, self.noise_percentage])
            # noisy_binary_data.rename(columns={column: f"{column}_noisy_{self.noise_percentage}"}, inplace=True)

    #     combined_data = X.copy()
    #     #print(combined_data)
    #     #print(noisy_numeric_data)
    #    # print(noisy_binary_data)
    #     combined_data[noisy_numeric_data.columns] = noisy_numeric_data
    #     combined_data[noisy_categorical_data.columns] = noisy_categorical_data

        # print(combined_data)
        return df_combined_data
    