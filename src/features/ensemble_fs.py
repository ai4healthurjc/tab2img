from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
import pandas as pd
import os.path as path
from pathlib import Path
from skrebate import ReliefF, MultiSURF
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from itertools import chain
import argparse
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from scipy.stats import gmean
import matplotlib
import os
import random
import numpy as np
from itertools import chain
import logging
import coloredlogs
from typing import Tuple

matplotlib.logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def get_bootstrap_sample(df_data: pd.DataFrame,
                         labels: np.array,
                         seed_value: int
                         ) -> Dict:
    """
    This function takes as input the data and labels and returns
    a bootstrap sample of the data, as well as its out-of-bag (OOB) data

    INPUTS:
    - data is a 2-dimensional numpy.ndarray where rows are examples and columns are features
    - labels is a 1-dimansional numpy.ndarray giving the label of each example in data

    OUPUT:
    - a dictionnary where:
          - key 'bootData' gives a 2-dimensional numpy.ndarray which is a bootstrap sample of data
          - key 'bootLabels' is a 1-dimansional numpy.ndarray giving the label of each example in bootData
          - key 'OOBData' gives a 2-dimensional numpy.ndarray the OOB examples
          - key 'OOBLabels' is a 1-dimansional numpy.ndarray giving the label of each example in OOBData
    """

    df_data_copy = df_data.copy()
    df_data = df_data_copy.to_numpy()
    n_samples, n_features = df_data.shape
    if n_samples != len(labels):
        raise ValueError('The data and labels should have a same number of rows.')

    # Set seed value
    np.random.seed(seed_value)

    ind = np.random.choice(range(n_samples), size=n_samples, replace=True)
    oo_bind = np.setdiff1d(range(n_samples), ind, assume_unique=True)
    boot_data = df_data[ind, ] #taking the ind row, all the columns
    boot_labels = labels[ind]
    oob_data = df_data[oo_bind, ]
    oob_labels = labels[oo_bind]

    dict_result = {
        'id_boot': seed_value,
        'boot_data': boot_data,
        'boot_labels': boot_labels,
        'OOBData': oob_data,
        'OOBLabels': oob_labels
     }

    return dict_result


class FSMethod(object):

    def __init__(self, return_scores=False, seed_value=3242):
        self.df_features = None
        self.selected_feature_indices = []
        self.selected_feature_names = []
        self.verbose = False
        self.df_feature_scores = None
        self.return_scores = return_scores
        self.seed_value = seed_value

    def fit(self, df_features: pd.DataFrame, y_label: np.array, k: int,
            list_categorical_vars: list, list_numerical_vars: list,
            verbose):

        self.df_features = df_features

    def extract_features(self, k=None):
        x_features = self.df_features.values
        if k is not None and k < len(self.selected_feature_indices):
            x_features_filtered = x_features[:, self.selected_feature_indices[:k]]
            df_features_selected = self.df_features.iloc[:, self.selected_feature_indices[:k]]
        # else:
            # x_features_filtered = x_features[:, self.selected_feature_indices]

        df_features_selected = self.df_features.iloc[:, self.selected_feature_indices]

        if self.return_scores:
            return df_features_selected, self.df_feature_scores
        else:
            return df_features_selected

    def _validate_categorical_columns(self, data, categorical_columns):
        """
        Check whether ``discrete_columns`` exists in ``train_data``.
        Args:
            data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            categorical_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(data, pd.DataFrame):
            invalid_columns = set(categorical_columns) - set(data.columns)
        elif isinstance(data, np.ndarray):
            invalid_columns = []
            for column in categorical_columns:
                if column < 0 or column >= data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')


def get_fs_method(fs_method_name: str):
    """

    Parameters
    ----------
    fs_method_name : str
        We define the filtered feature selection method we are going to use

    Returns
    -------
    Feature selection method: class
        Return a feature selection method from the folder fs

    """
    if fs_method_name == 'relief':
        return Relieved_F(return_scores=True)
    elif fs_method_name == 'mrmr':
        return MRMR(return_scores=True)
    elif fs_method_name == 'multisurf':
        return MultiSurf(return_scores=True)
    else:
        logger.info('This method has not been developed yet')
        return None
    
    
class MRMR(FSMethod):

    def __init__(self, return_scores):
        super().__init__(return_scores)
        self.selected_feature_names = None

    def fit(self,
            df_features: pd.DataFrame,
            y_label: np.unique,
            k: int,
            list_categorical_vars: list,
            list_numerical_vars: list,
            verbose=False
            ):

        self.df_features = df_features
        v_col_names = df_features.columns.values

        tuple_scores_df_filtered_features = mrmr_classif(X=df_features,
                                                         y=y_label,
                                                         K=k,
                                                         cat_features=list_categorical_vars,
                                                         cat_encoding='target',
                                                         return_scores=self.return_scores,
                                                         show_progress=False
                                                         )
        
        # logger.info(tuple_scores_df_filtered_features)

        v_scores_features = np.array(tuple_scores_df_filtered_features[1])
        v_scores_features_sorted = v_scores_features[v_scores_features.argsort()[::-1]]
        v_selected_feature_names_sorted = v_col_names[v_scores_features.argsort()][::-1]
        list_selected_features_index = [df_features.columns.get_loc(var_name) for var_name in v_selected_feature_names_sorted]

        self.selected_feature_indices = list_selected_features_index[:k]
        self.selected_feature_names = list(v_selected_feature_names_sorted[:k])

        m_scores = np.c_[v_selected_feature_names_sorted, v_scores_features_sorted]
        df_feature_scores = pd.DataFrame(m_scores, columns=['var_name', 'score'])

        self.df_feature_scores = df_feature_scores


class Relieved_F(FSMethod):

    def __init__(self, return_scores):
        super().__init__(return_scores)

    def fit(self,
            df_features: pd.DataFrame,
            y_label: np.array,
            k_features: int,
            list_categorical_vars: list,
            list_numerical_vars: list,
            verbose=False):

        # logger.info('Training filter fs method: {}'.format())

        self.df_features = df_features
        x_features = df_features.values

        relief = ReliefF(n_features_to_select=k_features)
        relief.fit(x_features, y_label)
        x_selected_features, list_selected_features_indices, scores_sorted = relief.transform(x_features)
                
        list_selected_feature_names = df_features.iloc[:, list_selected_features_indices].columns.values

        df_feature_scores = pd.DataFrame(np.zeros((len(list_selected_features_indices), 2)), columns=['var_name', 'score'])
        df_feature_scores['var_name'] = np.array(list_selected_feature_names)
        df_feature_scores['score'] = scores_sorted
        
        self.selected_feature_indices = list_selected_features_indices[:k_features]
        self.selected_feature_names = list_selected_feature_names[:k_features]
        self.df_feature_scores = df_feature_scores


class MultiSurf(FSMethod):

    def __init__(self, return_scores):
        super().__init__(return_scores)

    def fit(self, df_features: pd.DataFrame, y_label: np.array, k: int,
            list_categorical_vars: list, list_numerical_vars: list,
            verbose=False):

        self.df_features = df_features
        x_features = df_features.values

        surf = MultiSURF(n_features_to_select=k)
        surf.fit(x_features, y_label)
        x_selected_features, list_selected_features_indices, scores_sorted = surf.transform(x_features)

        list_selected_feature_names = df_features.iloc[:, list_selected_features_indices].columns.values

        df_feature_scores = pd.DataFrame(np.zeros((len(list_selected_features_indices), 2)), columns=['var_name', 'score'])
        df_feature_scores['var_name'] = np.array(list_selected_feature_names)
        df_feature_scores['score'] = scores_sorted

        self.selected_feature_indices = list_selected_features_indices[:k]
        self.selected_feature_names = list_selected_feature_names[:k]
        self.df_feature_scores = df_feature_scores


def set_seed_reproducibility(seed):
    """
    Generating the seed to control randomness

    Parameters
    ----------
    seed : int
        Seed to be used throughout the process

    Returns
    -------
    Seed
    """
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)


def perform_ensemble_fs(features_scaled: pd.DataFrame,
                        y_res: np.array,
                        fs_method_name,
                        type_aggregation,
                        v_feature_names,
                        list_vars_categorical,
                        list_vars_numerical,
                        n_boots=50,
                        n_jobs=1
                        ):

    Z_selected, Z_scores = compute_zmatrix_bootstrap(features_scaled,
                                                     y_res,
                                                     fs_method_name,
                                                     v_feature_names,
                                                     list_vars_categorical,
                                                     list_vars_numerical,
                                                     n_boots,
                                                     n_jobs=n_jobs)

    df_ensemble_voting_sorted, df_ensemble_mean_sorted = run_ensemble_agg(features_scaled,
                                                                          Z_selected,
                                                                          Z_scores,
                                                                          type_aggregation)

    # print(df_ensemble_voting_sorted)

    df_voting_sorted = df_ensemble_voting_sorted.sort_values(by="score", ascending=False)

    return df_voting_sorted


def compute_zmatrix_bootstrap(df_features: pd.DataFrame,
                              y_label: np.array,
                              fs_method_name: str,
                              v_feature_names: list,
                              list_vars_categorical: list,
                              list_vars_numerical: list,
                              M: int = 50,
                              n_jobs=1
                              ) -> Tuple[np.array, np.array]:
    """
    Computes the bootstrap and the ensembles and aggregating with both voting and mean approaches.

    Parameters
    ----------
    df_features : pd.DataFrame
        A DataFrame without the label obtained from the dataset.
    y_label : pd.Series
        A Series containing the label.
    fs_method_name : str
        A string containing the feature selection method we are going to use.# methods_to_use = ['relieved-f', 'mrmr', 'multisurf']
    v_feature_names : list
        A list containing the name of the features.
    list_vars_categorical: list
        A list of categorical vars
    list_vars_numerical: list
        A list of numerical vars
    M: int, optional
        Number of partitions with replacements (bootstraps). The default is 50.
    n_jobs: int (Default: 1)
        The number of jobs used for training

    Returns
    -------
    Z : np.array
        A tensor containing the features chosen (1 or 0) over the M partitions throughout the different columns.
    Z_scores: np.array
        A matrix containing the score of the features chosen over the M partitions throughout the different columns.
    """
    n_samples, n_features = df_features.shape
    k_values = range(n_features)
    y_label = y_label.reshape(n_samples)
    Z_selected = np.zeros((len(k_values), M, n_features), dtype=np.int8)
    Z_scores = np.zeros((len(k_values), M, n_features), dtype=np.single)

    list_n_boots = list(range(M))

    list_dict_boot_data = [get_bootstrap_sample(df_features, y_label, id_boot) for id_boot in list_n_boots]

    # n_jobs = cpu_count() if n_jobs == -1 else min(cpu_count(), n_jobs)

    list_dict_fs_result = Parallel(n_jobs=n_jobs)(
        delayed(train_fs_method_by_bootstrap)(dict_boot_data, fs_method_name, v_feature_names, list_vars_categorical, list_vars_numerical) for dict_boot_data in list_dict_boot_data
    )

    for id_bootstrap in list_n_boots:

        dict_fs_result = list(filter(lambda dict_fs: dict_fs['id_boot'] == id_bootstrap, list_dict_fs_result))[0]

        for k_features in k_values:
            df_k_features = dict_fs_result['k_features_all'].iloc[:, :k_features + 1]
            df_k_scores = dict_fs_result['k_scores_all'].iloc[:k_features + 1, :]
            v_col_names_k_features = df_k_features.columns.values
            top_k_scores = df_k_scores['score'].values
            top_k = dict_fs_result['boot_data'].columns.get_indexer(v_col_names_k_features)
            Z_selected[k_features, id_bootstrap, top_k] = 1
            Z_scores[k_features, id_bootstrap, top_k] = top_k_scores
            # logger.info('boot: {}, k_values: {}, scores: {}'.format(id_bootstrap, k_features + 1, list(top_k_scores)))

    return Z_selected, Z_scores


def train_fs_method_by_bootstrap(dict_boot_data, fs_method_name, v_feature_names, list_vars_categorical, list_vars_numerical):

    logger.info('Train fs: {} with boot: {}'.format(fs_method_name, dict_boot_data['id_boot']))

    if isinstance(dict_boot_data['boot_data'], np.ndarray):
        dict_boot_data['boot_data'] = pd.DataFrame(np.squeeze(dict_boot_data['boot_data']), columns=v_feature_names)

    n_samples, n_features = dict_boot_data['boot_data'].shape

    fs_method = get_fs_method(fs_method_name)
    fs_method.fit(dict_boot_data['boot_data'],
                  dict_boot_data['boot_labels'],
                  n_features,
                  list_vars_categorical,
                  list_vars_numerical
                  )
    df_k_features_all, df_k_scores_all = fs_method.extract_features()

    dict_fs_result = {
        'id_boot': dict_boot_data['id_boot'],
        'boot_data': dict_boot_data['boot_data'],
        'k_features_all': df_k_features_all,
        'k_scores_all': df_k_scores_all
    }

    return dict_fs_result


def get_agg_func(agg_type):
    if agg_type == 'median':
        agg_func = np.median
    elif agg_type == 'max':
        agg_func = np.max
    elif agg_type == 'gmean':
        agg_func = gmean(axis=1)
    elif agg_type == 'mean':
        agg_func = np.mean
    else:
        agg_func = np.mean

    return agg_func


def run_ensemble_agg(df_features,
                     Z_selected,
                     Z_scores,
                     agg_func='all'
                     ):
    """

    Parameters
    ----------
    df_features
    Z_selected
    Z_scores
    agg_type

    Returns
    -------
    ensemble_voting : np.array
        An array containing the number of votes per features chosen.
    ensemble_mean: np.array
        An array containing the mean of the scores of relevance for each feature selection method per features chosen.

    """

    n_samples, n_features = df_features.shape
    v_col_names = df_features.columns.values

    agg_func = get_agg_func(agg_func)

    list_v_agg_selected = []
    list_v_agg_scores = []
    for k_features in range(n_features):
        v_agg_k_features_selected = np.sum(Z_selected[k_features, :, :], axis=0)
        v_agg_k_features_scores = agg_func(Z_scores[k_features, :, :], axis=0)
        list_v_agg_selected.append(v_agg_k_features_selected)
        list_v_agg_scores.append(v_agg_k_features_scores)

    v_ensemble_voting = np.sum(np.asarray(list_v_agg_selected), axis=0)
    v_ensemble_simple = agg_func(np.asarray(list_v_agg_scores), axis=0)

    df_ensemble_voting = pd.DataFrame({'var_name': v_col_names, 'score': v_ensemble_voting})
    df_ensemble_simple = pd.DataFrame({'var_name': v_col_names, 'score': v_ensemble_simple})

    df_ensemble_voting_sorted = df_ensemble_voting.sort_values(by=['score'], ascending=False)
    df_ensemble_simple_sorted = df_ensemble_simple.sort_values(by=['score'], ascending=False)

    return df_ensemble_voting_sorted, df_ensemble_simple_sorted


def compute_stability(Z,
                      n_features,
                      stability_measure=None
                      ):
    """
    It computes the stability and the error while computing it (confidence intervals)
    by using a bootstrap tensor (Z).

    Parameters
    ----------
    Z : TYPE
        DESCRIPTION.
    n_features : TYPE
        DESCRIPTION.
    stability_measure : str, optional
        Stability measure to be used. The default is None, given that
        we are using the one from Sarah Nogueira.

    Returns
    -------
    stabilities : TYPE
        DESCRIPTION.
    stab_err : TYPE
        DESCRIPTION.

    """

    k_values = range(n_features)
    stabilities = np.zeros(len(k_values))
    stab_err = np.zeros(len(k_values))

    for j in range(len(k_values)):
        res = st.compute_confidence_intervals(Z[j, ], alpha=0.05)
        stabilities[j] = res['stability']
        stab_err[j] = stabilities[j] - res['lower']
    stabilities[-1] = 1 # maximum stability     
    stab_err[-1] = 0 # minimum error 
    return stabilities, stab_err


def get_bbdd_name(bbdd_name):
    # LIST_BBDD_HIGH_DIM_BINARY = [BBDD_PHISING, BBDD_ADS, BBDD_PARKINSON, BBDD_HOSPITAL, BBDD_SPEEDDATING]
    # LIST_BBDD_LOW_DIM_BINARY = [BBDD_CRX, BBDD_HEPATITIS, BBDD_IONOS, BBDD_SAHEART, BBDD_DIABETES, BBDD_GERMAN]
    # if bbdd_name in consts.LIST_BBDD_LOW_DIM_BINARY or bbdd_name in consts.LIST_BBDD_HIGH_DIM_BINARY:
    # return bbdd_name
    return bbdd_name


def save_dataframe_results(df_results_current):
    results_name = df_results_current.attrs['results_name']
    path_results_file = Path.joinpath(consts.PATH_PROJECT_RESULTS, '{}.csv'.format(results_name))

    if path.exists(str(path_results_file)):
        df_results_history = pd.read_csv(str(path_results_file))
        df_results_all = pd.concat([df_results_history, df_results_current], ignore_index=True)
        df_results_all.to_csv(str(path_results_file), index=False)
    else:
        df_results_current.to_csv(str(path_results_file), index=False)


def get_normalized_data_features(x_features, v_feature_names, scaler_num, scaler_cat):
    df_features = pd.DataFrame(x_features, columns=v_feature_names)

    df_features[list_vars_numerical] = scaler_num.fit_transform(df_features[list_vars_numerical].values)
    df_features[list_vars_categorical] = scaler_cat.fit_transform(df_features[list_vars_categorical].values)

    return df_features


# def parse_arguments(parser):
#     #parser.add_argument('--bbdd', default='diabetes', type=str)
#     parser.add_argument('--n_boots', default=100, type=int)
#     parser.add_argument('--n_jobs', default=-1, type=int)
#     #parser.add_argument('--estimator', default='dt', type=str)
#     parser.add_argument('--fs', default='relief', type=str)
#     parser.add_argument('--agg_func', default='mean', type=str)
#     # parser.add_argument('--weight', default='0.0', type=str)
#     return parser.parse_args()


# parser = argparse.ArgumentParser(description='Process representations.')
# args = parse_arguments(parser)
# # weight = atof(args.weight)

# bbdd_name = get_bbdd_name(args.bbdd)
# x_features, y_label, v_feature_names, list_vars_categorical, list_vars_numerical = load_preprocessed_dataset(bbdd_name)

# logger.info('vars num: {}'.format(list_vars_numerical))
# logger.info('vars cat: {}'.format(list_vars_categorical))

# df_features = get_normalized_data_features(x_features, v_feature_names, MinMaxScaler(), MinMaxScaler())
# set_seed_reproducibility(1002)
# columns = df_features.columns

# df_features_original = df_features.copy()

# Compute matrix Z_selected and Z_scores

def get_categorical_numerical_names(df_data: pd.DataFrame) -> (list, list):

    df_info = identify_type_features(df_data)
    list_numerical_vars = list(df_info[df_info['type'] == 'c'].index)
    list_categorical_vars = list(df_info[df_info['type'] == 'd'].index)

    return list_categorical_vars, list_numerical_vars


def identify_type_features(df, discrete_threshold=10, debug=False):
    """
    Categorize every feature/column of df as discrete or continuous according to whether or not the unique responses
    are numeric and, if they are numeric, whether or not there are fewer unique renponses than discrete_threshold.
    Return a dataframe with a row for each feature and columns for the type, the count of unique responses, and the
    count of string, number or null/nan responses.
    """
    counts = []
    string_counts = []
    float_counts = []
    null_counts = []
    types = []
    for col in df.columns:
        responses = df[col].unique()
        counts.append(len(responses))
        string_count, float_count, null_count = 0, 0, 0
        for value in responses:
            try:
                val = float(value)
                if not math.isnan(val):
                    float_count += 1
                else:
                    null_count += 1
            except:
                try:
                    val = str(value)
                    string_count += 1
                except:
                    print('Error: Unexpected value', value, 'for feature', col)

        string_counts.append(string_count)
        float_counts.append(float_count)
        null_counts.append(null_count)
        types.append('d' if len(responses) < discrete_threshold or string_count > 0 else 'c')

    feature_info = pd.DataFrame({'count': counts,
                                 'string_count': string_counts,
                                 'float_count': float_counts,
                                 'null_count': null_counts,
                                 'type': types}, index=df.columns)
    if debug:
        print(f'Counted {sum(feature_info["type"] == "d")} discrete features and {sum(feature_info["type"] == "c")} continuous features')

    return feature_info
