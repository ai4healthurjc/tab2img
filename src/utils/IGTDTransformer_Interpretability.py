from scipy.stats import spearmanr, rankdata
from scipy.spatial.distance import pdist, squareform
import gower
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
import time
import _pickle as cp
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import rankdata
# from util.loader import compute_corr_mixed_dataset, compute_corr_mixed_dataset_2
from utils.plotter import plot_corr_mixed_matrix
from utils.dissimilarity import compute_corr_mixed_dataset
from matplotlib import image
import cv2
import seaborn as sns
import matplotlib.colors as mcolors
import re
from matplotlib.colors import to_rgba
from gower import gower_matrix


class IGTD_Transformer(BaseEstimator, TransformerMixin):

    def __init__(self, num_row=4, num_col=4, save_image_size=3, max_step=1000, val_step=100, result_dir=None,
                 numericas=None, categoricas=None, error='abs',save_path='', type_corr='pearson'):
        self.num_row = num_row
        self.num_col = num_col
        self.save_image_size = save_image_size
        self.max_step = max_step
        self.val_step = val_step
        self.result_dir = result_dir
        self.numericas = numericas
        self.categoricas = categoricas
        self.error = error
        self.save_path = save_path
        self.type_corr = type_corr

    def fit(self, data, y=None):
        # Realiza cualquier proceso de entrenamiento necesario.
        return self

    def min_max_transform(self,data):
        '''
        This function does a linear transformation of each feature, so that the minimum and maximum values of a
        feature are 0 and 1, respectively.

        Input:
        data: an input data array with a size of [n_sample, n_feature]
        Return:
        norm_data: the data array after transformation
        '''

        norm_data = np.empty(data.shape)
        norm_data.fill(np.nan)
        for i in range(data.shape[1]):
            v = data[:, i].copy()
            if np.max(v) == np.min(v):
                norm_data[:, i] = 0
            else:
                v = (v - np.min(v)) / (np.max(v) - np.min(v))
                norm_data[:, i] = v
        return norm_data

    def generate_feature_distance_ranking(self, data):
        
        num_vars = data.shape[1]

        if not self.categoricas and not self.numericas:
            raise ValueError("Not categorical nor continuous variables found!")
        
        # if not self.numericas:
        #     corr = np.corrcoef(np.transpose(data[self.categoricas]))
        # elif not self.categoricas:
        if self.type_corr == 'pearson':
            corr = spearmanr(data).correlation
        else:
            corr = compute_corr_mixed_dataset(data, self.categoricas, self.save_path)
        
        corr = 1 - corr
        corr = np.around(a=corr, decimals=10)

        if isinstance(corr, np.ndarray):
            tril_id = np.tril_indices(num_vars, k=-1)
            rank = rankdata(corr[tril_id])
        else:
            tril_id = np.tril_indices(num_vars, k=-1)
            rank = rankdata(corr.values[tril_id])
            
        ranking = np.zeros((num_vars, num_vars))
        ranking[tril_id] = rank
        ranking = ranking + np.transpose(ranking)
        print(ranking)
        print(corr)

        return ranking, corr

    def generate_matrix_distance_ranking(self, data):
        '''
        This function calculates the ranking of distances between all pairs of entries in a matrix of size num_r by num_c.

        Input:
        num_r: number of rows in the matrix
        num_c: number of columns in the matrix
        method: method used to calculate distance. Can be 'Euclidean' or 'Manhattan'.

        Return:
        coordinate: num_r * num_c by 2 matrix giving the coordinates of elements in the matrix.
        ranking: a num_r * num_c by num_r * num_c matrix giving the ranking of pair-wise distance.

        '''
        
        if not self.categoricas and not self.numericas:
            raise ValueError("No se encontraron características binarias ni numéricas en el conjunto de datos.")
        
        if not self.categoricas:
            method = 'Euclidean'
        elif not self.numericas:
            method = 'Manhattan'
        else:
            method = 'Gower'

        # generate the coordinates of elements in a matrix
        # Genera el ranking de distancias en función del método
        if method == 'Euclidean':
            # Calcula la matriz de distancias euclidianas
            euclidean_matrix = squareform(pdist(data, metric='euclidean'))
            euclidean_matrix = np.max(euclidean_matrix) - euclidean_matrix
            euclidean_matrix = euclidean_matrix / np.max(euclidean_matrix)
            cord_dist = euclidean_matrix

        elif method == 'Manhattan':
            # Calcula la matriz de distancias Manhattan
            manhattan_matrix = squareform(pdist(data, metric='cityblock'))
            manhattan_matrix = np.max(manhattan_matrix) - manhattan_matrix
            manhattan_matrix = manhattan_matrix / np.max(manhattan_matrix)
            cord_dist = manhattan_matrix

        elif method == 'Gower':
            cat_vector=[False]*len(data.iloc[0])
            col=list(data.columns)
            for e in self.categoricas:
                index=col.index(e)
                cat_vector[index] = True
            cord_dist = gower_matrix(data, cat_features= cat_vector)
        # generate the ranking based on distance
        num = self.num_row * self.num_col
        tril_id = np.tril_indices(num, k=-1)
        rank = rankdata(cord_dist[tril_id])
        ranking = np.zeros((num, num))
        ranking[tril_id] = rank
        ranking = ranking + np.transpose(ranking)

        # generate the coordinates of elements in a matrix
        for r in range(self.num_row):
            if r == 0:
                coordinate = np.transpose(np.vstack((np.zeros(self.num_col), range(self.num_col))))
            else:
                ones_array = np.ones(self.num_col) * r
                range_array = np.arange(self.num_col)
                transposed_range = np.transpose(np.vstack((ones_array, range_array)))
                coordinate = np.vstack((coordinate, transposed_range))
                #coordinate = np.vstack((coordinate, np.transpose(np.vstack((np.ones(self.num_col) * r, range(self.num_col)))))

        coordinate = np.int64(coordinate)

        return (coordinate[:, 0], coordinate[:, 1]), ranking

    def IGTD_absolute_error(self, source, target, max_step=1000,
                            switch_t=0, val_step=50,
                            min_gain=0.00001, random_state=1,
                            save_folder=None, file_name=''):
        '''
        This function switches the order of rows (columns) in the source ranking matrix to make it similar to the target
        ranking matrix. In each step, the algorithm randomly picks a row that has not been switched with others for
        the longest time and checks all possible switch of this row, and selects the switch that reduces the
        dissimilarity most. Dissimilarity (i.e. the error) is the summation of absolute difference of
        lower triangular elements between the rearranged source ranking matrix and the target ranking matrix.

        Input:
        source: a symmetric ranking matrix with zero diagonal elements.
        target: a symmetric ranking matrix with zero diagonal elements. 'source' and 'target' should have the same size.
        max_step: the maximum steps that the algorithm should run if never converges.
        switch_t: the threshold to determine whether switch should happen
        val_step: number of steps for checking gain on the objective function to determine convergence
        min_gain: if the objective function is not improved more than 'min_gain' in 'val_step' steps,
            the algorithm terminates.
        random_state: for setting random seed.
        save_folder: a path to save the picture of source ranking matrix in the optimization process.
        file_name: a string as part of the file names for saving results

        Return:
        index_record: indices to rearrange the rows(columns) in source obtained the optimization process
        err_record: error obtained in the optimization process
        run_time: the time at which each step is completed in the optimization process
        '''
        # print(source.shape)
        # print(target.shape)
        np.random.RandomState(seed=random_state)
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        os.mkdir(save_folder)

        source = source.copy()
        # source = np.pad(source, pad_width=10, mode='minimum')
        # target = np.pad(target, pad_width=10, mode='minimum')

        num = source.shape[0]
        tril_id = np.tril_indices(num, k=-1)
        index = np.array(range(num))
        index_record = np.empty((max_step + 1, num))
        index_record.fill(np.nan)
        index_record[0, :] = index.copy()

        # calculate the error associated with each row
        err_v = np.empty(num)
        err_v.fill(np.nan)
        for i in range(num):
            err_v[i] = np.sum(np.abs(source[i, 0:i] - target[i, 0:i])) + \
                    np.sum(np.abs(source[(i + 1):, i] - target[(i + 1):, i]))

        step_record = -np.ones(num)
        err_record = [np.sum(abs(source[tril_id] - target[tril_id]))]
        pre_err = err_record[0]
        t1 = time.time()
        run_time = [0]

        for s in range(max_step):
            delta = np.ones(num) * np.inf

            # randomly pick a row that has not been considered for the longest time
            idr = np.where(step_record == np.min(step_record))[0]
            ii = idr[np.random.permutation(len(idr))[0]]

            for jj in range(num):
                if jj == ii:
                    continue

                if ii < jj:
                    i = ii
                    j = jj
                else:
                    i = jj
                    j = ii

                err_ori = err_v[i] + err_v[j] - np.abs(source[j, i] - target[j, i])

                err_i = np.sum(np.abs(source[j, :i] - target[i, :i])) + \
                        np.sum(np.abs(source[(i + 1):j, j] - target[(i + 1):j, i])) + \
                        np.sum(np.abs(source[(j + 1):, j] - target[(j + 1):, i])) + np.abs(source[i, j] - target[j, i])
                err_j = np.sum(np.abs(source[i, :i] - target[j, :i])) + \
                        np.sum(np.abs(source[i, (i + 1):j] - target[j, (i + 1):j])) + \
                        np.sum(np.abs(source[(j + 1):, i] - target[(j + 1):, j])) + np.abs(source[i, j] - target[j, i])
                err_test = err_i + err_j - np.abs(source[i, j] - target[j, i])

                delta[jj] = err_test - err_ori

            delta_norm = delta / pre_err
            id = np.where(delta_norm <= switch_t)[0]

            if len(id) > 0:
                jj = np.argmin(delta)

                # Update the error associated with each row
                if ii < jj:
                    i = ii
                    j = jj
                else:
                    i = jj
                    j = ii
                for k in range(num):
                    if k < i:
                        err_v[k] = err_v[k] - np.abs(source[i, k] - target[i, k]) - np.abs(source[j, k] - target[j, k]) + \
                                np.abs(source[j, k] - target[i, k]) + np.abs(source[i, k] - target[j, k])
                    elif k == i:
                        err_v[k] = np.sum(np.abs(source[j, :i] - target[i, :i])) + \
                        np.sum(np.abs(source[(i + 1):j, j] - target[(i + 1):j, i])) + \
                        np.sum(np.abs(source[(j + 1):, j] - target[(j + 1):, i])) + np.abs(source[i, j] - target[j, i])
                    elif k < j:
                        err_v[k] = err_v[k] - np.abs(source[k, i] - target[k, i]) - np.abs(source[j, k] - target[j, k]) + \
                                np.abs(source[k, j] - target[k, i]) + np.abs(source[i, k] - target[j, k])
                    elif k == j:
                        err_v[k] = np.sum(np.abs(source[i, :i] - target[j, :i])) + \
                        np.sum(np.abs(source[i, (i + 1):j] - target[j, (i + 1):j])) + \
                        np.sum(np.abs(source[(j + 1):, i] - target[(j + 1):, j])) + np.abs(source[i, j] - target[j, i])
                    else:
                        err_v[k] = err_v[k] - np.abs(source[k, i] - target[k, i]) - np.abs(source[k, j] - target[k, j]) + \
                                np.abs(source[k, j] - target[k, i]) + np.abs(source[k, i] - target[k, j])

                # switch rows i and j
                ii_v = source[ii, :].copy()
                jj_v = source[jj, :].copy()
                source[ii, :] = jj_v
                source[jj, :] = ii_v
                ii_v = source[:, ii].copy()
                jj_v = source[:, jj].copy()
                source[:, ii] = jj_v
                source[:, jj] = ii_v
                err = delta[jj] + pre_err

                # update rearrange index
                t = index[ii]
                index[ii] = index[jj]
                index[jj] = t

                # update step record
                step_record[ii] = s
                step_record[jj] = s
            else:
                # error is not changed due to no switch
                err = pre_err

                # update step record
                step_record[ii] = s

            err_record.append(err)
            print('Step ' + str(s) + ' err: ' + str(err))
            print('Source matrix:')
            print(source)
            print('Target matrix:')
            print(target)
            index_record[s + 1, :] = index.copy()
            run_time.append(time.time() - t1)
            # print('Step ' + str(s) + ' err: ' + str(err))
            # index_record[s + 1, :] = index.copy()
            # run_time.append(time.time() - t1)

            if s > val_step:
                if np.sum((err_record[-val_step - 1] - np.array(err_record[(-val_step):])) / err_record[-val_step - 1] >= min_gain) == 0:
                    break

            pre_err = err

        # index_record = index_record[:len(err_record), :].astype(np.int)
        index_record = index_record[:len(err_record), :].astype(int)

        if save_folder is not None:
            pd.DataFrame(index_record).to_csv(
                save_folder + '/' + file_name + '_index.txt',
                header=False, index=False, sep='\t'
            )
            pd.DataFrame(np.transpose(np.vstack((err_record, np.array(range(s + 2))))),
                         columns=['error', 'steps']).to_csv(save_folder + '/' + file_name + '_error_and_step.txt',
                                                            header=True, index=False, sep='\t')
            pd.DataFrame(np.transpose(np.vstack((err_record, run_time))), columns=['error', 'run_time']).to_csv(
                save_folder + '/' + file_name + '_error_and_time.txt', header=True, index=False, sep='\t')

        return index_record, err_record, run_time

    def IGTD_square_error(self, source, target, max_step=1000, switch_t=0, val_step=50, min_gain=0.00001,
                          random_state=1, save_folder=None, file_name=''):
        '''
        This function switches the order of rows (columns) in the source ranking matrix to make it similar to the target
        ranking matrix. In each step, the algorithm randomly picks a row that has not been switched with others for
        the longest time and checks all possible switch of this row, and selects the switch that reduces the
        dissimilarity most. Dissimilarity (i.e. the error) is the summation of squared difference of
        lower triangular elements between the rearranged source ranking matrix and the target ranking matrix.

        Input:
        source: a symmetric ranking matrix with zero diagonal elements.
        target: a symmetric ranking matrix with zero diagonal elements. 'source' and 'target' should have the same size.
        max_step: the maximum steps that the algorithm should run if never converges.
        switch_t: the threshold to determine whether switch should happen
        val_step: number of steps for checking gain on the objective function to determine convergence
        min_gain: if the objective function is not improved more than 'min_gain' in 'val_step' steps,
            the algorithm terminates.
        random_state: for setting random seed.
        save_folder: a path to save the picture of source ranking matrix in the optimization process.
        file_name: a string as part of the file names for saving results

        Return:
        index_record: ordering index to rearrange the rows(columns) in 'source' in the optimization process
        err_record: the error history in the optimization process
        run_time: the time at which each step is finished in the optimization process
        '''

        np.random.RandomState(seed=random_state)
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        os.mkdir(save_folder)

        source = source.copy()
        num = source.shape[0]
        tril_id = np.tril_indices(num, k=-1)
        index = np.array(range(num))
        index_record = np.empty((max_step + 1, num))
        index_record.fill(np.nan)
        index_record[0, :] = index.copy()

        # calculate the error associated with each row
        err_v = np.empty(num)
        err_v.fill(np.nan)

        for i in range(num):
            err_v[i] = np.sum(np.square(source[i, 0:i] - target[i, 0:i])) + np.sum(np.square(source[(i + 1):, i] - target[(i + 1):, i]))

        step_record = -np.ones(num)
        err_record = [np.sum(np.square(source[tril_id] - target[tril_id]))]
        pre_err = err_record[0]
        t1 = time.time()
        run_time = [0]

        for s in range(max_step):
            delta = np.ones(num) * np.inf

            # randomly pick a row that has not been considered for the longest time
            idr = np.where(step_record == np.min(step_record))[0]
            ii = idr[np.random.permutation(len(idr))[0]]

            for jj in range(num):
                if jj == ii:
                    continue

                if ii < jj:
                    i = ii
                    j = jj
                else:
                    i = jj
                    j = ii

                err_ori = err_v[i] + err_v[j] - np.square(source[j, i] - target[j, i])

                err_i = np.sum(np.square(source[j, :i] - target[i, :i])) + \
                        np.sum(np.square(source[(i + 1):j, j] - target[(i + 1):j, i])) + \
                        np.sum(np.square(source[(j + 1):, j] - target[(j + 1):, i])) + np.square(source[i, j] - target[j, i])

                err_j = np.sum(np.square(source[i, :i] - target[j, :i])) + \
                        np.sum(np.square(source[i, (i + 1):j] - target[j, (i + 1):j])) + \
                        np.sum(np.square(source[(j + 1):, i] - target[(j + 1):, j])) + np.square(source[i, j] - target[j, i])

                err_test = err_i + err_j - np.square(source[i, j] - target[j, i])

                delta[jj] = err_test - err_ori

            delta_norm = delta / pre_err
            id = np.where(delta_norm <= switch_t)[0]
            if len(id) > 0:
                jj = np.argmin(delta)

                # Update the error associated with each row
                if ii < jj:
                    i = ii
                    j = jj
                else:
                    i = jj
                    j = ii
                for k in range(num):
                    if k < i:
                        err_v[k] = err_v[k] - np.square(source[i, k] - target[i, k]) - np.square(source[j, k] - target[j, k]) + \
                                np.square(source[j, k] - target[i, k]) + np.square(source[i, k] - target[j, k])
                    elif k == i:
                        err_v[k] = np.sum(np.square(source[j, :i] - target[i, :i])) + \
                            np.sum(np.square(source[(i + 1):j, j] - target[(i + 1):j, i])) + \
                            np.sum(np.square(source[(j + 1):, j] - target[(j + 1):, i])) + np.square(source[i, j] - target[j, i])
                    elif k < j:
                        err_v[k] = err_v[k] - np.square(source[k, i] - target[k, i]) - np.square(source[j, k] - target[j, k]) + \
                                np.square(source[k, j] - target[k, i]) + np.square(source[i, k] - target[j, k])
                    elif k == j:
                        err_v[k] = np.sum(np.square(source[i, :i] - target[j, :i])) + \
                            np.sum(np.square(source[i, (i + 1):j] - target[j, (i + 1):j])) + \
                            np.sum(np.square(source[(j + 1):, i] - target[(j + 1):, j])) + np.square(source[i, j] - target[j, i])
                    else:
                        err_v[k] = err_v[k] - np.square(source[k, i] - target[k, i]) - np.square(source[k, j] - target[k, j]) + \
                                np.square(source[k, j] - target[k, i]) + np.square(source[k, i] - target[k, j])

                # switch rows i and j
                ii_v = source[ii, :].copy()
                jj_v = source[jj, :].copy()
                source[ii, :] = jj_v
                source[jj, :] = ii_v
                ii_v = source[:, ii].copy()
                jj_v = source[:, jj].copy()
                source[:, ii] = jj_v
                source[:, jj] = ii_v
                err = delta[jj] + pre_err

                # update rearrange index
                t = index[ii]
                index[ii] = index[jj]
                index[jj] = t

                # update step record
                step_record[ii] = s
                step_record[jj] = s
            else:
                # error is not changed due to no switch
                err = pre_err

                # update step record
                step_record[ii] = s

            err_record.append(err)
            print('Step ' + str(s) + ' err: ' + str(err))
            index_record[s + 1, :] = index.copy()
            run_time.append(time.time() - t1)

            if s > val_step:
                if np.sum((err_record[-val_step - 1] - np.array(err_record[(-val_step):])) / err_record[
                    -val_step - 1] >= min_gain) == 0:
                    break

            pre_err = err

        index_record = index_record[:len(err_record), :].astype(np.int)
        if save_folder is not None:
            pd.DataFrame(index_record).to_csv(
                save_folder + '/' + file_name + '_index.txt',
                header=False, index=False, sep='\t'
            )
            pd.DataFrame(np.transpose(np.vstack((err_record, np.array(range(s + 2))))),
                        columns=['error', 'steps']).to_csv(save_folder + '/' + file_name + '_error_and_step.txt',
                                                            header=True, index=False, sep='\t')
            pd.DataFrame(np.transpose(np.vstack((err_record, run_time))), columns=['error', 'run_time']).to_csv(
                save_folder + '/' + file_name + '_error_and_time.txt', header=True, index=False, sep='\t')

        return index_record, err_record, run_time

    def IGTD(self, source, target, err_measure='abs', max_step=1000, switch_t=0, val_step=50, min_gain=0.00001,
             random_state=1, save_folder=None, file_name=''):
        '''
        This is just a wrapper function that wraps the two search functions using different error measures.
        '''

        if err_measure == 'abs':
            index_record, err_record, run_time = self.IGTD_absolute_error(source=source,
                                                                          target=target,
                                                                          max_step=max_step,
                                                                          switch_t=switch_t,
                                                                          val_step=val_step,
                                                                          min_gain=min_gain,
                                                                          random_state=random_state,
                                                                          save_folder=save_folder,
                                                                          file_name=file_name
                                                                          )
        if err_measure == 'squared':
            index_record, err_record, run_time = self.IGTD_square_error(source=source,
                                                                        target=target,
                                                                        max_step=max_step,
                                                                        switch_t=switch_t,
                                                                        val_step=val_step,
                                                                        min_gain=min_gain,
                                                                        random_state=random_state,
                                                                        save_folder=save_folder,
                                                                        file_name=file_name
                                                                        )

        return index_record, err_record, run_time

    def generate_image_data(self, data, index, num_row, num_column, coord,
                            original_feature_names, image_folder=None, image_folder_interpretability=None, file_name=''):
        if isinstance(data, pd.DataFrame):
            samples = data.index.map(str)
            data = data.values
        else:
            samples = [str(i) for i in range(data.shape[0])]

        feature_names_subset = [original_feature_names[i] for i in index]

        if os.path.exists(image_folder):
            shutil.rmtree(image_folder)
        os.mkdir(image_folder)

        if os.path.exists(image_folder_interpretability):
            shutil.rmtree(image_folder_interpretability)
        os.mkdir(image_folder_interpretability)

        data_2 = data.copy()
        data_2 = data_2[:, index]
        max_v = np.max(data_2)
        min_v = np.min(data_2)
        data_2 = 255 - (data_2 - min_v) / (max_v - min_v) * 255  # So that black means high value
        image_data = np.empty((num_row, num_column, data_2.shape[0]))
        image_data.fill(np.nan)

        # Obtener los nombres de las características originales en el conjunto index
        original_feature_names_subset = [original_feature_names[i] for i in index]
        print(original_feature_names_subset)

        num_features = len(original_feature_names_subset)
        print(num_features)

        if num_features <= 25:
            N = 5
        elif 25 < num_features < 50:
            N = 10
        else:
            N = 20

        # Paletas de colores base
        base_cmaps = ['Purples', 'Reds', 'Blues', 'Oranges', 'Greens']

        # Crear una lista de colores combinando colores de las paletas base
        colors = np.concatenate([plt.get_cmap(name)(np.linspace(0.1, 0.9, N)) for name in base_cmaps])

        # Crear un nuevo mapa de colores con la lista de colores
        cmap = mcolors.ListedColormap(colors)
        colors_dict = {}

        for i, feature in enumerate(original_feature_names_subset):
            if '_' in feature:
                components = feature.split('_')
                prefix = components[0]
                # Verificar si el siguiente elemento no es "gaussian" o "power"
                if len(components) > 1 and components[1] not in ['gaussian', 'power']:
                    prefix += '_' + components[1]
            else:
                prefix = feature
                print(prefix)

            colors_dict[prefix] = cmap(i)

        print(colors_dict)

        legend_fig, legend_ax = plt.subplots(figsize=(2, 2))
        # legend_elements_all = [] 

        for i in range(data_2.shape[0]):
            data_i = np.empty((num_row, num_column), dtype=object)
            data_i_image = np.empty((num_row, num_column))
            data_i_image.fill(np.nan)
            data_i.fill('')  # Inicializa con cadenas vacías
            data_i_image[coord]=data_2[i,:]
            data_i_copy=data_i.copy()
            image_data[:, :, i] = data_i_image
            # Variable para almacenar los valores originales de las características
            feature_values = np.zeros(len(coord[0]))

            for j in range(len(coord[0])):
                row, col = coord[0][j], coord[1][j]
                feature_name = original_feature_names[index[j]]
                feature_value = data_2[i, j]

                if 'power' in feature_name.lower():
                    marker = 'o'  
                elif 'gaussian' in feature_name.lower():
                    marker = 'o'
                else:
                    marker = 'o'  

                data_i[row, col] = marker
                data_i_copy[row, col] = feature_name
                # Almacena el valor original de la característica
                feature_values[j] = feature_value

            if image_folder is not None:
                fig = plt.figure(figsize=(num_row, num_column))
                plt.imshow(data_i_image, cmap='gray', vmin=0, vmax=255)
                #plt.axis('scaled')
                plt.axis('off')  # Esto desactiva los ejes x e y
                #print(data_i)
                #print('----')
                plt.savefig(fname=image_folder + '/' + file_name + '_' + samples[i] + '_image.png', bbox_inches='tight')
                plt.close(fig)
                #cv2.imwrite(image_folder + '/' + file_name + '_' + samples[i] + '_image.png', data_i)

                pd.DataFrame(image_data[:, :, i], index=None, columns=None).to_csv(
                    image_folder + '/' + file_name + '_' + samples[i] + '_data.txt',
                    # '{}/{}_{}_data.txt'.format(image_folder, file_name, samples[i]),
                    header=None,
                    index=None,
                    sep='\t'
                )

                fig, ax = plt.subplots(figsize=(num_row, num_column))
                im = ax.imshow(feature_values.reshape((num_row, num_column)), cmap='gray', vmin=0, vmax=255)

                for row in range(num_row):
                    for col in range(num_column):
                        marker = data_i[row, col]
                        feature_name = data_i_copy[row, col]
                        if marker != '':
                            if '_' in feature_name:
                                components = feature_name.split('_')
                                prefix = components[0]
                                # Verificar si el siguiente elemento no es "gaussian" o "power"
                                if len(components) > 1 and components[1] not in ['gaussian', 'power']:
                                    prefix += '_' + components[1]
                                    print(prefix)
                            else:
                                prefix = feature_name
                                print(prefix)

                            print(feature_name)
                            print(colors_dict.get(prefix))
                            ax.scatter(col, row, marker=marker, color=colors_dict.get(prefix), edgecolors='black', linewidth=1, s=400)
               
                            # Verifica si la variable es ruidosa antes de agregar el texto con el porcentaje de ruido
                            if 'power' in feature_name.lower() or 'gaussian' in feature_name.lower():
                                # Extrae el porcentaje de ruido del nombre de la característica
                                # percentage_of_noise = float(feature_name.split('_')[-1])
                                # print(percentage_of_noise)
                                print(feature_name)
                                # Agrega texto en el centro del marcador con el porcentaje de ruido
                                # ax.text(col, row, f"{percentage_of_noise:.2f}", color='black', ha='center', va='center', fontsize=8)
                plt.axis('off')
                plt.savefig(fname=image_folder_interpretability + '/' + file_name + '_' + samples[i] + '_image.png', bbox_inches='tight')
                plt.close(fig)

                legend_labels = {}
                for i, feature in enumerate(original_feature_names_subset):
                    if '_' in feature:
                        components = feature.split('_')
                        prefix = components[0]
                        # Verificar si el siguiente elemento no es "gaussian" o "power"
                        if len(components) > 1 and components[1] not in ['gaussian', 'power']:
                            prefix += '_' + components[1]
                    else:
                        prefix = feature

                    # Crea leyendas por conjunto de características originales y ruidosas con el mismo prefijo
                    legend_labels[feature] = ('o', colors_dict.get(prefix, colors_dict.get(feature, 'black')))

                print(legend_labels)
                filtered_legend_labels = {name: (marker, color) for name, (marker, color) in legend_labels.items() if 'gaussian' not in name.lower() and 'power' not in name.lower()}
                print(filtered_legend_labels)
                
                 #Obtiene los nombres de las características originales presentes en la leyenda
                present_originals = set()

                for feature_name in filtered_legend_labels.keys():
                    if '_' in feature_name:
                        components = feature_name.split('_')
                        prefix = components[0]
                        # Verificar si el siguiente elemento no es "gaussian" o "power"
                        if len(components) > 1 and components[1] not in ['gaussian', 'power']:
                            prefix += '_' + components[1]
                    else:
                        prefix = feature_name

                    present_originals.add(prefix)
                    
                all_originals = set()
                for feature_name in original_feature_names_subset:
                    if '_' in feature_name:
                        components = feature_name.split('_')
                        prefix = components[0]
                        # Verificar si el siguiente elemento no es "gaussian" o "power"
                        if len(components) > 1 and components[1] not in ['gaussian', 'power']:
                            prefix += '_' + components[1]
                    else:
                        prefix = feature_name

                    all_originals.add(prefix)
                print(present_originals)
                # Determina las características originales faltantes
                missing_originals = all_originals - present_originals
                print(missing_originals)
                # Agrega las características originales faltantes a la leyenda con colores predeterminados
                for missing_original in missing_originals:
                    filtered_legend_labels[missing_original] = ('o', colors_dict[missing_original])
                print(filtered_legend_labels)
                # Crea elementos de leyenda
                legend_elements = [plt.Line2D([0], [0],
                                              marker=marker,
                                              color=color,
                                              label=label,
                                              markerfacecolor=color,
                                              markersize=8,
                                              markeredgewidth=1,
                                              markeredgecolor='black')
                                for label, (marker, color) in filtered_legend_labels.items()]

                #ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
                print(legend_elements)
                
                # pd.DataFrame(image_data[:, :, i], index=None, columns=None).to_csv(image_folder + '/' + file_name + '_'
                #             + samples[i] + '_data.txt', header=None, index=None, sep='\t')
        print(legend_elements)
        legend_ax.legend(handles=legend_elements, loc='center', bbox_to_anchor=(1, 1), fontsize=8)
        legend_ax.axis('off')

        # Guarda la figura de la leyenda
        legend_fig.savefig(fname=image_folder_interpretability + '/legend.png', bbox_inches='tight')
        plt.close(legend_fig) 

        return image_data, samples

    def table_to_image(self, norm_d, scale, save_image_size, max_step, val_step, normDir,
                       error, switch_t=0, min_gain=0.00001):
        '''
        This function converts tabular data into images using the IGTD algorithm. 

        Input:
        norm_d: a 2D array or data frame, which is the tabular data. Its size is n_samples by n_features
        scale: a list of two positive integers. The number of pixel rows and columns in the image representations,
            into which the tabular data will be converted.
        fea_dist_method: a string indicating the method used for calculating the pairwise distances between features, 
            for which there are three options.
            'Pearson' uses the Pearson correlation coefficient to evaluate the similarity between features.
            'Spearman' uses the Spearman correlation coefficient to evaluate the similarity between features.
            'set' uses the Jaccard index to evaluate the similarity between features that are binary variables.
        image_dist_method: a string indicating the method used for calculating the distances between pixels in image.
            It can be either 'Euclidean' or 'Manhattan'.
        save_image_size: size of images (in inches) for saving visual results.
        max_step: the maximum number of iterations that the IGTD algorithm will run if never converges.
        val_step: the number of iterations for determining algorithm convergence. If the error reduction rate is smaller than 
            min_gain for val_step iterations, the algorithm converges.
        normDir: a string indicating the directory to save result files.
        error: a string indicating the function to evaluate the difference between feature distance ranking and pixel
            distance ranking. 'abs' indicates the absolute function. 'squared' indicates the square function.
        switch_t: the threshold on error change rate. Error change rate is
            (error after feature swapping - error before feature swapping) / error before feature swapping.
            In each iteration, if the smallest error change rate resulted from all possible feature swappings
            is not larger than switch_t, the feature swapping resulting in the smallest error change rate will
            be performed. If switch_t <= 0, the IGTD algorithm monotonically reduces the error during optimization.
        min_gain: if the error reduction rate is not larger than min_gain for val_step iterations, the algorithm converges.
        
        Return:
        This function does not return any variable, but saves multiple result files, which are the following
        1.  Results.pkl stores the original tabular data, the generated image data, and the names of samples. The generated
            image data is a 3D numpy array. Its size is [number of pixel rows in image, number of pixel columns in image,
            number of samples]. The range of values is [0, 255]. Small values in the array actually correspond to high
            values in the tabular data.
        2.  Results_Auxiliary.pkl stores the ranking matrix of pairwise feature distances before optimization,
            the ranking matrix of pairwise pixel distances, the coordinates of pixels when concatenating pixels
            row by row from image to form the pixel distance ranking matrix, error in each iteration,
            and time (in seconds) when completing each iteration.
        3.  original_feature_ranking.png shows the feature distance ranking matrix before optimization.
        4.  image_ranking.png shows the pixel distance ranking matrix.
        5.  error_and_runtime.png shows the change of error vs. time during the optimization process.
        6.  error_and_iteration.png shows the change of error vs. iteration during the optimization process.
        7.  optimized_feature_ranking.png shows the feature distance ranking matrix after optimization.
        8.  data folder includes two image data files for each sample. The txt file is the image data in matrix format.
            The png file shows the visualization of image data.
        '''

        if os.path.exists(normDir):
            shutil.rmtree(normDir)
        os.mkdir(normDir)

        ranking_feature, corr = self.generate_feature_distance_ranking(data=norm_d)
        original_feature_names = norm_d.columns.tolist() if isinstance(norm_d, pd.DataFrame) else None

        fig = plt.figure()
        plt.imshow(np.max(ranking_feature) - ranking_feature, cmap='gray', interpolation='nearest')
        plt.savefig(fname=normDir + '/original_feature_ranking.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        coordinate, ranking_image = self.generate_matrix_distance_ranking(data=norm_d)

        fig = plt.figure()
        plt.imshow(np.max(ranking_image) - ranking_image, cmap='gray', interpolation='nearest')
        plt.savefig(fname=normDir + '/image_ranking.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        index, err, time = self.IGTD(source=ranking_feature, target=ranking_image,
                                     err_measure=error, max_step=max_step, switch_t=switch_t, val_step=val_step,
                                     min_gain=min_gain, random_state=1,
                                     save_folder=normDir + '/' + error, file_name=''
                                     )

        fig = plt.figure()
        plt.plot(time, err)
        plt.savefig(fname=normDir + '/error_and_runtime.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        fig = plt.figure()
        plt.plot(range(len(err)), err)
        plt.savefig(fname=normDir + '/error_and_iteration.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        min_id = np.argmin(err)

        ranking_feature_random = ranking_feature[index[min_id, :], :]
        ranking_feature_random = ranking_feature_random[:, index[min_id, :]]

        # fig = plt.figure(figsize=(save_image_size, save_image_size))
        fig = plt.figure()
        plt.imshow(np.max(ranking_feature_random) - ranking_feature_random, cmap='gray', interpolation='nearest')
        plt.savefig(fname=normDir + '/optimized_feature_ranking.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        data, samples = self.generate_image_data(data=norm_d,
                                                 index=index[min_id, :],
                                                 num_row=scale[0],
                                                 num_column=scale[1],
                                                 coord=coordinate,
                                                 original_feature_names=original_feature_names,
                                                 image_folder=normDir + '/data',
                                                 image_folder_interpretability=normDir + '/data_interpretability',
                                                 file_name='')

        output = open(normDir + '/Results.pkl', 'wb')
        cp.dump(norm_d, output)
        cp.dump(data, output)
        cp.dump(samples, output)
        output.close()

        output = open(normDir + '/Results_Auxiliary.pkl', 'wb')
        cp.dump(ranking_feature, output)
        cp.dump(ranking_image, output)
        cp.dump(coordinate, output)
        cp.dump(err, output)
        cp.dump(time, output)
        output.close()

    def transform(self, data):
        return self.table_to_image(data,
                                   [self.num_row, self.num_col],
                                   self.save_image_size,
                                   self.max_step,
                                   self.val_step,
                                   self.result_dir,
                                   self.error
                                   )
