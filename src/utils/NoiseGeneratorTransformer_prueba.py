import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NoiseDataGenerator(BaseEstimator, TransformerMixin):

    def __init__(self, numeric_vars=None, categorical_vars=None, generation_count=0, sigma=None, noise_power=None):
        self.numeric_vars = numeric_vars
        self.categorical_vars = categorical_vars
        self.sigma = sigma 
        self.noise_power = noise_power
        self.generation_count = generation_count

    def _generate_gaussian_noise(self, X):

        # Apply Gaussian noise to numeric variables
        X_numeric = X[self.numeric_vars].copy()
        noisy_numeric_data = pd.DataFrame()

        for col in X_numeric.columns:

            gaussian_noise = np.random.normal(0, self.sigma, X_numeric.shape[0])
            noisy_col_name = f"{col}_gaussian_{round(self.sigma, 2)}_{self.generation_count}"
            noisy_numeric_data[noisy_col_name] = X_numeric[col] + gaussian_noise

        return noisy_numeric_data
    
#DO NOT REMOVE
    # def _generate_swap_noise(self, X):
    #     noisy_categorical_data = X[self.categorical_vars].copy()
    #     num_samples, num_features = noisy_categorical_data.shape
    #     num_swap = int(num_samples * self.noise_power)

    #     for _ in range(num_swap):
    #         i, j = np.random.choice(num_samples, 2, replace=False)
    #         noisy_categorical_data.iloc[i], noisy_categorical_data.iloc[j] = (
    #             noisy_categorical_data.iloc[j].copy(),
    #             noisy_categorical_data.iloc[i].copy(),
    #         )

    #     for col in noisy_categorical_data.columns:
    #         noisy_categorical_data.rename(columns={col: f"{col}_power_{round(self.noise_power, 2)}_{self.generation_count}"}, inplace=True)

    #     return noisy_categorical_data
    def _generate_swap_noise(self, X):
        df_noisy_categorical_data = X[self.categorical_vars].copy()
        n_samples, n_categorical_features = df_noisy_categorical_data.shape
        # num_samples = len(df_noisy_categorical_data)
        idx = range(n_samples)
        n_row_changes = int(n_samples * self.noise_power)
        num_changes = int(n_samples * self.noise_power)

        # DAVID
        # for pos_cat_feature in range(n_categorical_features):
        #     col_vals = np.random.permutation(df_noisy_categorical_data[:, pos_cat_feature])
        #     swap_idx = np.random.choice(idx, size=n_row_changes)
        #     df_noisy_categorical_data.iloc[swap_idx, pos_cat_feature] = np.random.choice(col_vals, size=n_row_changes)

        for _ in range(num_changes):
            i = np.random.choice(n_samples)
            available_categories = np.unique(df_noisy_categorical_data)
            current_category = df_noisy_categorical_data.iloc[i]
            if current_category.iloc[0] in available_categories:
                available_categories_filtered = available_categories[available_categories != current_category.iloc[0]]
                if len(available_categories_filtered) > 0:
                    new_category = np.random.choice(available_categories_filtered)
                df_noisy_categorical_data.iloc[i] = new_category

        for col in df_noisy_categorical_data.columns:
            df_noisy_categorical_data.rename(columns={col: f"{col}_power_{round(self.noise_power, 2)}_{self.generation_count}"}, inplace=True)

        return df_noisy_categorical_data

    def _generate_salt_pepper_noise(self, X):
        
        # prob =self.noise_power
        # noisy_binary_data = X[self.categorical_vars].copy()

        # for column in self.categorical_vars:
        #     rnd = np.random.rand(noisy_binary_data.shape[0])
        #     noisy_binary_data[column][(rnd < prob)] = 0
        #     noisy_binary_data[column][(rnd > 1 - prob)] = 1
        #     noisy_binary_data.rename(columns={column: f"{column}_noisy_{self.noise_power}_{self.generation_count}"}, inplace=True)

        # return noisy_binary_data
    
        
        noisy_binary_data = X[self.categorical_vars].copy()
        for column in self.categorical_vars:
            noisy_binary_data[column] = np.random.choice([0, 1], size=len(X), p=[1 - self.noise_power, self.noise_power])
            noisy_binary_data.rename(columns={column: f"{column}_power_{round(self.noise_power, 2)}_{self.generation_count}"}, inplace=True)
        return noisy_binary_data



    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df_combined_data = X.copy()

        for col in self.numeric_vars:
            noisy_numeric_data = self._generate_gaussian_noise(X)
            df_combined_data = pd.concat((df_combined_data, noisy_numeric_data), axis=1)

        for col in self.categorical_vars:
            noisy_categorical_data = self._generate_swap_noise(X)
            # noisy_categorical_data = self._generate_salt_pepper_noise(X)
            df_combined_data = pd.concat((df_combined_data, noisy_categorical_data), axis=1)

        return df_combined_data
