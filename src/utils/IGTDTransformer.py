import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import shutil
import time
import _pickle as cp
import gower
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import rankdata
from utils.dissimilarity import compute_corr_mixed_dataset, compute_corr_mixed_dataset_2
# from gudmm import ditance_dependency_mixed_matrix


class IGTDTransformer(BaseEstimator, TransformerMixin):

    def __init__(self,
                 num_row=4,
                 num_col=4,
                 save_image_size=3,
                 max_step=1000,
                 val_step=100,
                 result_dir=None,
                 numericas=None,
                 categoricas=None,
                 error='abs'
                 ):

        self.num_row = num_row
        self.num_col = num_col
        self.save_image_size = save_image_size
        self.max_step = max_step
        self.val_step = val_step
        self.result_dir = result_dir
        self.numericas = numericas
        self.categoricas = categoricas
        self.error = error

    def fit(self, data, y=None):
        return self

    def perform_min_max_transform(self, data):
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
        # print(len(self.numericas))
        # print(len(self.categoricas))

        if not self.categoricas and not self.numericas:
            raise ValueError("No se encontraron características binarias ni numéricas en el conjunto de datos.")
        
        if not self.numericas:
            corr = np.corrcoef(np.transpose(data[self.categoricas]))
        elif not self.categoricas:
            corr = spearmanr(data[self.numericas]).correlation
        else:
            corr = compute_corr_mixed_dataset_2(data, self.categoricas, self.numericas)

        corr = 1 - corr
        corr = np.around(a=corr, decimals=10)

        if isinstance(corr, np.ndarray):
            tril_id = np.tril_indices(num_vars, k=-1)
            rank = rankdata(corr[tril_id])
        else: # dataframe
            tril_id = np.tril_indices(num_vars, k=-1)
            rank = rankdata(corr.values[tril_id])
            
        ranking = np.zeros((num_vars, num_vars))
        ranking[tril_id] = rank
        ranking = ranking + np.transpose(ranking)

        return ranking, corr

    def generate_matrix_distance_ranking(self, data):
        '''
        This function calculates the ranking of distances between all pairs of entries in a matrix of size Nr x Nc

        Input:
        Nr: number of rows in the matrix
        Nc: number of columns in the matrix
        method: method used to calculate distance. For instance, 'Euclidean', 'Manhattan'.

        Return:
        coordinate: num_r * num_c by 2 matrix giving the coordinates of elements in the matrix.
        ranking: a num_r * num_c by num_r * num_c matrix giving the ranking of pair-wise distance.

        '''
        
        if not self.categoricas and not self.numericas:
            raise ValueError("No se encontraron características binarias ni numéricas en el conjunto de datos.")
        
        if not self.numericas:
            method = 'Euclidean'
        elif not self.categoricas:
            method = 'Manhattan'
        else:
            method = 'Gower'

        # generate the coordinates of elements in a matrix
        # Genera el ranking de distancias en función del método

        if method == 'Euclidean':
            euclidean_matrix = squareform(pdist(data, metric='euclidean'))
            euclidean_matrix = np.max(euclidean_matrix) - euclidean_matrix
            euclidean_matrix = euclidean_matrix / np.max(euclidean_matrix)
            cord_dist = euclidean_matrix

        elif method == 'Manhattan':
            manhattan_matrix = squareform(pdist(data, metric='cityblock'))
            manhattan_matrix = np.max(manhattan_matrix) - manhattan_matrix
            manhattan_matrix = manhattan_matrix / np.max(manhattan_matrix)
            cord_dist = manhattan_matrix

        elif method == 'Gower':
            # cord_dist = gower.gower_matrix(np.transpose(data))
            cord_dist = gower.gower_matrix(data)
            # dist = ditance_dependency_mixed_matrix(data, no_f_cont, no_f_ord, DM)

        # Generate the ranking based on distance
        num = self.num_row * self.num_col
        tril_id = np.tril_indices(num, k=-1)
        rank = rankdata(cord_dist[tril_id])
        # print(num)
        ranking = np.zeros((num, num))
        ranking[tril_id] = rank
        ranking = ranking + np.transpose(ranking)
        # print(ranking.sh)

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

    def IGTD_absolute_error(self,
                            source, # ranking feature
                            target, # ranking pixel
                            max_step=1000,
                            switch_t=0,
                            val_step=50,
                            min_gain=0.00001,
                            random_state=1,
                            save_folder=None,
                            file_name=''
                            ):
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
            err_v[i] = np.sum(np.abs(source[i, 0:i] - target[i, 0:i])) + np.sum(np.abs(source[(i + 1):, i] - target[(i + 1):, i]))

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
                        err_v[k] = \
                            err_v[k] - np.abs(source[i, k] - target[i, k]) - np.abs(source[j, k] - target[j, k]) + \
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
            index_record[s + 1, :] = index.copy()
            run_time.append(time.time() - t1)

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

    def IGTD_square_error(self, source, target,
                          max_step=1000, switch_t=0,
                          val_step=50, min_gain=0.00001,
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
            err_v[i] = np.sum(np.square(source[i, 0:i] - target[i, 0:i])) + \
                    np.sum(np.square(source[(i + 1):, i] - target[(i + 1):, i]))

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

    def IGTD(self,
             source,
             target,
             err_measure='abs',
             max_step=1000,
             switch_t=0,
             val_step=50,
             min_gain=0.00001,
             random_state=1,
             save_folder=None,
             file_name=''
             ):

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

    def generate_image_data(self, data, index, num_row, num_column, coord, image_folder=None, file_name=''):
        '''
        This function generates the data in image format according to rearrangement indices. It saves the data
        sample-by-sample in both txt files and image files

        Input:
        data: original tabular data, 2D array or data frame, n_samples by n_features
        index: indices of features obtained through optimization, according to which the features can be
            arranged into a num_r by num_c image.
        num_row: number of rows in image
        num_column: number of columns in image
        coord: coordinates of features in the image/matrix
        image_folder: directory to save the image and txt data files. If none, no data file is saved
        file_name: a string as a part of the file names to save data

        Return:
        image_data: the generated data, a 3D numpy array. The third dimension is across samples. The range of values
            is [0, 255]. Small values actually indicate high values in the original data.
        samples: the names of indices of the samples
        '''

        if isinstance(data, pd.DataFrame):
            # samples = data.index.map(np.str)
            samples = data.index.map(str)
            data = data.values
        else:
            samples = [str(i) for i in range(data.shape[0])]

        if os.path.exists(image_folder):
            shutil.rmtree(image_folder)
        os.mkdir(image_folder)

        data_2 = data.copy()
        data_2 = data_2[:, index]
        max_v = np.max(data_2)
        min_v = np.min(data_2)
        data_2 = 255 - (data_2 - min_v) / (max_v - min_v) * 255 # So that black means high value

        image_data = np.empty((num_row, num_column, data_2.shape[0]))
        # print(num_row, num_column)
        image_data.fill(np.nan)
        for i in range(data_2.shape[0]):
            data_i = np.empty((num_row, num_column))
            data_i.fill(np.nan)
            data_i[coord] = data_2[i, :]
            image_data[:, :, i] = data_i
            #print(data_i.shape)
            if image_folder is not None:
                #print(image_data)
                fig = plt.figure(figsize=(num_row, num_column))
                plt.imshow(data_i, cmap='gray', vmin=0, vmax=255)
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

        return image_data, samples

    def table_to_image(self, norm_d, scale, save_image_size,
                       max_step, val_step, normDir,
                       error, switch_t=0, min_gain=0.00001
                       ):
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

        print('holi')

        norm_d = norm_d.iloc[:, 1:-1].copy()

        ranking_feature, corr = self.generate_feature_distance_ranking(data=norm_d)

        fig = plt.figure()
        plt.imshow(np.max(ranking_feature) - ranking_feature, cmap='gray', interpolation='nearest')
        plt.savefig(fname=normDir + '/original_feature_ranking.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        coordinate, ranking_image = self.generate_matrix_distance_ranking(data=norm_d)

        fig = plt.figure()
        plt.imshow(np.max(ranking_image) - ranking_image, cmap='gray', interpolation='nearest')
        plt.savefig(fname=normDir + '/image_ranking.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        index, err, time = self.IGTD(source=ranking_feature,
                                     target=ranking_image,
                                     err_measure=error,
                                     max_step=max_step,
                                     switch_t=switch_t,
                                     val_step=val_step,
                                     min_gain=min_gain,
                                     random_state=1,
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
        # ranking_feature_random = ranking_feature[index[min_id, :], :]
        # ranking_feature_random = ranking_feature_random[:, index[min_id, :]]

        # fig = plt.figure()
        # plt.imshow(np.max(ranking_feature_random) - ranking_feature_random, cmap='gray', interpolation='nearest')
        # plt.savefig(fname=normDir + '/optimized_feature_ranking.png', bbox_inches='tight', pad_inches=0)
        # plt.close(fig)

        data, samples = self.generate_image_data(data=norm_d,
                                                 index=index[min_id, :],
                                                 num_row=scale[0],
                                                 num_column=scale[1],
                                                 coord=coordinate,
                                                 image_folder=normDir + '/data',
                                                 file_name=''
                                                 )

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
                                   self.save_image_size, self.max_step,
                                   self.val_step, self.result_dir, self.error
                                   )
