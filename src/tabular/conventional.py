import numpy as np
import pandas as pd
from collections import Counter
from typing import List
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import utils.consts as cons
# from clfs.evaluator import compute_classification_prestations, get_metric_classification

import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def balance_partition(df_x, y, seed_value):

    rus = RandomUnderSampler(random_state=seed_value)
    x_resampled, y_resampled = rus.fit_resample(df_x, y)

    logger.info('# samples per class: {}'.format(sorted(Counter(y_resampled).items())))

    return x_resampled, y_resampled


def train_with_several_partitions_by_clf(df_x_features: pd.DataFrame,
                                         y_label: np.array,
                                         estimator_name: str,
                                         fs_method: str,
                                         k_features: int,
                                         scoring_estimator: str = 'roc_auc',
                                         as_frame: bool = True,
                                         verbose: bool = False
                                         ) -> List[dict]:

    logger.info('Training with x {}'.format(df_x_features.shape))

    list_total_metrics = []
    list_estimators = cons.LIST_CONVENTIONAL_ESTIMATORS

    if estimator_name in list_estimators:
        list_dict_metrics = train_several_partitions(estimator_name,
                                                     fs_method,
                                                     df_x_features,
                                                     y_label,
                                                     k_features=k_features,
                                                     scoring_estimator=scoring_estimator,
                                                     verbose=verbose)

        list_total_metrics.extend(list_dict_metrics)
    else:
        ValueError('Conventional ML model for tabular data is not found!')

    # if as_frame:
    #     return pd.DataFrame(list_total_metrics)
    # else:
    #     return list_total_metrics

    return list_total_metrics


def train_clf_with_selected_features(df_filtered_features: pd.DataFrame,
                                     y_label: np.array,
                                     bbdd_name: str,
                                     estimator_name: str = 'dt',
                                     as_frame: bool = False,
                                     scoring_estimator: str = 'roc_auc',
                                     ):

    v_ranked_feature_names = df_filtered_features.columns.values

    logger.info('vars-selected: {}'.format(list(v_ranked_feature_names)))

    k_features = df_filtered_features.shape[1]

    list_total_metrics = train_with_several_partitions_by_clf(df_filtered_features,
                                                              y_label,
                                                              estimator_name,
                                                              fs_method='na',
                                                              k_features=k_features,
                                                              scoring_estimator=scoring_estimator,
                                                              as_frame=False,
                                                              verbose=True
                                                              )

    if as_frame:
        df_metrics = pd.DataFrame(list_total_metrics)
        return df_metrics
    else:
        return list_total_metrics


def get_clf_hyperparams(classifier: str, n_train_samples: int, seed_value: int):
    selected_clf = None
    param_grid = {}

    if classifier == 'knn':
        selected_clf = KNeighborsClassifier()

        param_grid = {
            'n_neighbors': range(1, 20, 2),
            # 'metric': 'hamming'
        }

    elif classifier == 'svm':
        selected_clf = SVC()

        param_grid = {
            'decision_function_shape': 'ovo',
            'kernel': ['rbf', 'poly'],
            'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
            'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]
        }

    elif classifier == 'dt':

        lenght_15_percent_val = int(0.1 * n_train_samples)
        lenght_20_percent_val = int(0.2 * n_train_samples)

        selected_clf = DecisionTreeClassifier(random_state=seed_value)

        param_grid = {
            'max_depth': range(2, 20, 1),
            'criterion': ['gini', 'entropy'],
            'min_samples_split': [2, 3, 4, 5, 6],
            # 'min_samples_split': range(2, lenght_20_percent_val, 1)
        }

    elif classifier == 'reglog':
        selected_clf = LogisticRegression()
        param_grid = {
            'penalty': ['l1'],
            'C': [1e-4, 1e-2, 1, 5, 10, 20]
        }

    elif classifier == 'lasso':

        selected_clf = Lasso(max_iter=10000)

        param_grid = {
            'alpha': np.logspace(-6, 3, 10)
        }

    elif classifier == 'rf':
        selected_clf = RandomForestClassifier(random_state=seed_value)

        param_grid = {
            'n_estimators': [5, 10, 20],
            'max_depth': range(1, 16, 2),
        }

    return selected_clf, param_grid


def perform_clf(estimator_name: str,
                seed_value: int,
                scoring_estimator,
                x_train: np.array,
                y_train: np.array,
                x_test: np.array,
                y_test: np.array,
                verbose=False
                ) -> dict:

    estimator_model, param_grid = get_clf_hyperparams(estimator_name, x_train.shape[0], seed_value)

    logger.info('estimator: {}, params: {}'.format(estimator_name, param_grid))

    return compute_clf_metrics(x_train, y_train, x_test, y_test, estimator_model, param_grid, scoring_estimator, verbose)


def compute_clf_metrics(x_train: np.array,
                        y_train: np.array,
                        x_test: np.array,
                        y_test: np.array,
                        clf, param_grid,
                        scoring_clf,
                        verbose=False
                        ) -> dict:

    n_classes = len(set(y_test))
    scoring_gridcv = get_metric_classification(scoring_clf, n_classes)

    logger.info('Scoring gridcv: {} '.format(scoring_gridcv))

    grid_cv = GridSearchCV(clf, param_grid=param_grid, scoring=scoring_gridcv, cv=5, return_train_score=True)
    grid_cv.fit(x_train, y_train)

    # auc_knn_all_train = np.array(grid_cv.cv_results_['mean_train_score'])
    # auc_knn_all_val = np.array(grid_cv.cv_results_['mean_test_score'])
    # plot_grid(param_grid['n_neighbors'], auc_knn_all_train, auc_knn_all_val)

    logger.info('Best hyperparams: {}, best_score: {}'.format(grid_cv.best_params_, grid_cv.best_score_))

    best_clf = grid_cv.best_estimator_
    best_clf.fit(x_train, y_train)
    y_pred = best_clf.predict(x_test)

    # plot_feature_importance_by_clf(best_clf)

    dict_metrics = compute_classification_prestations(y_test, y_pred, np.unique(y_test), average='micro', verbose=True)

    logger.info(dict_metrics)

    return dict_metrics


def train_several_partitions(estimator_name, fs_method, df_x_features, y_label, k_features, scoring_estimator, verbose=False):

    list_dict_metrics = []

    for seed_value in range(1, 6, 1):

        x_train, x_test, y_train, y_test = train_test_split(df_x_features.values,
                                                            y_label,
                                                            stratify=y_label,
                                                            test_size=0.2,
                                                            random_state=seed_value
                                                            )

        dict_metrics = perform_clf(estimator_name, seed_value, scoring_estimator,
                                   x_train, y_train, x_test, y_test, verbose)

        logger.info(dict_metrics)

        list_dict_metrics.append(dict_metrics)

    list_metrics = list_dict_metrics[0].keys()
    list_dict_fm = []

    for metric_name in list_metrics:
        metric_values = [dict_metrics[metric_name] for dict_metrics in list_dict_metrics]

        list_dict_fm.append({'estimators': estimator_name,
                             'fs_method': fs_method,
                             'k_features': k_features,
                             'metric': metric_name,
                             'mean': np.mean(metric_values),
                             'std': np.std(metric_values)
                             })

    return list_dict_fm