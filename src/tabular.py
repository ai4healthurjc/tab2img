import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
# from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
import logging
import coloredlogs
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score, recall_score
from sklearn.linear_model import LogisticRegression
from tabpfn import TabPFNClassifier
import utils.consts as consts
from utils.loader import load_preprocessed_dataset, normalize_dataframe, load_dataset_augmented

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def compute_classification_performance(y_true: np.array,
                                       y_pred: np.array,
                                       y_prob) -> (float, float, float, float):
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        acc_val = accuracy_score(y_true, y_pred)
        specificity_val = tn / (tn + fp)
        recall_val = recall_score(y_true, y_pred)
        roc_val = roc_auc_score(y_true, y_pred)
    else:
        cm = confusion_matrix(y_true, y_pred)
        num_classes = cm.shape[0]
        specificity_val = 0

        for i in range(num_classes):
            true_negatives = sum(cm[j, j] for j in range(num_classes)) - cm[i, i]
            false_positives = sum(cm[:, i]) - cm[i, i]
            specificity_val += true_negatives / (true_negatives + false_positives)

        specificity_val /= num_classes

        acc_val = accuracy_score(y_true, y_pred)
        recall_val = recall_score(y_true, y_pred, average='macro')
        roc_val = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')

    return acc_val, specificity_val, recall_val, roc_val


def binarize_pred(y_pred):

    for i in range(len(y_pred)):
        if y_pred[i] > 0.5:
            y_pred[i] = 1.0
        else:
            y_pred[i] = 0.0
    return y_pred


def parse_arguments(parser):
    parser.add_argument('--dataset', default='fram', type=str)
    parser.add_argument('--categorical_encoding', default='count', type=str)
    parser.add_argument('--with_noise', default=0, type=int)
    parser.add_argument('--noise_type', default='homogeneous', type=str)
    parser.add_argument('--augmented', default=0, type=int)
    parser.add_argument('--type_sampling', default='over', type=str)
    parser.add_argument('--oversampler', default='ctgan', type=str)
    parser.add_argument('--n_seeds', default=5, type=int)
    parser.add_argument('--n_jobs', default=4, type=int)
    parser.add_argument('--classifier', default='tabpfn', type=str)
    parser.add_argument('--type_encoding', default='standard', type=str)
    return parser.parse_args()


cmd_parser = argparse.ArgumentParser(description='tab2img experiments')
args = parse_arguments(cmd_parser)

list_acc_values = []
list_specificity_values = []
list_recall_values = []
list_auc_values = []


for idx in consts.SEEDS:

    if not args.augmented:
        path_dataset, features, y_label = load_preprocessed_dataset(args.dataset, args.categorical_encoding)

    if args.augmented:
        if args.with_noise:
            file_name, X_train_scaled, X_test_scaled, Y_train, Y_test = load_dataset_augmented(
                args.with_noise, idx, args.dataset, args.type_sampling, args.oversampler, args.noise_type)
            rus = RandomUnderSampler(sampling_strategy='all', random_state=idx)
            X_test_scaled, Y_test = rus.fit_resample(X_test_scaled, Y_test)
        else:
            file_name, X_train_scaled, X_test_scaled, Y_train, Y_test = load_dataset_augmented(
                args.with_noise, idx, args.dataset, args.type_sampling, args.oversampler, noise_type=None)
            rus = RandomUnderSampler(sampling_strategy='all', random_state=idx)
            X_test_scaled, Y_test = rus.fit_resample(X_test_scaled, Y_test)
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(features, y_label, stratify=y_label, test_size=0.2, random_state=idx)

        X_train_scaled, X_test_scaled = normalize_dataframe(X_train, X_test, args.type_encoding)
        rus = RandomUnderSampler(sampling_strategy='all', random_state=idx)

        X_train_scaled, Y_train = rus.fit_resample(X_train_scaled, Y_train)
        print('Resampled dataset shape %s' % pd.DataFrame(Y_train).value_counts())
        print('Resampled dataset shape %s' % Y_test.value_counts())



    if args.classifier == 'svm':

        hyperparams_clf = {
            'C': np.logspace(-1, 1, 4)
        }

        # model_clf = LinearSVC(max_iter=100000, random_state=idx)
        model_clf = SVC(probability=True, max_iter=500, random_state=idx)

    elif args.classifier == 'dt':
        length_train = len(X_train_scaled)
        length_15_percent_val = int(0.15 * length_train)
        length_20_percent_val = int(0.20 * length_train)

        hyperparams_clf = {
            'max_depth': range(2, 8, 2),
            'min_samples_split': range(length_15_percent_val, length_20_percent_val),
            # 'min_samples_split': range(2, 20),
        }

        model_clf = DecisionTreeClassifier(random_state=idx)


    elif args.classifier == 'tabpfn':
        model_clf = TabPFNClassifier( device='cuda', random_state=idx)

        hyperparams_clf = {
            'n_estimators': [5, 10, 15],

        }


    elif args.classifier == 'knn':

        hyperparams_clf = {
            'n_neighbors': range(1, 15, 2)
        }        

        model_clf = KNeighborsClassifier()

    elif args.classifier == 'lasso':

        hyperparams_clf = {
            "C": np.logspace(-1.5, 0.4, 3),
            "penalty": ["l1"]
        }

        model_clf = LogisticRegression(solver='liblinear', max_iter=500, random_state=idx)

    grid_cv = GridSearchCV(estimator=model_clf, param_grid=hyperparams_clf, scoring='roc_auc', cv=5, n_jobs=args.n_jobs)
    grid_cv.fit(X_train_scaled, Y_train)

    clf_model = grid_cv.best_estimator_
    clf_model.fit(X_train_scaled, Y_train)
    y_pred = clf_model.predict(X_test_scaled)
    y_prob = clf_model.predict_proba(X_test_scaled)

    logger.info('Classifier {}, best_params: {}'.format(args.classifier, grid_cv.best_params_))

    if args.classifier == 'lasso':
        y_pred = binarize_pred(y_pred)

    acc_val, specificity_val, recall_val, roc_auc_val = compute_classification_performance(Y_test, y_pred, y_prob)

    list_acc_values.append(acc_val)
    list_specificity_values.append(specificity_val)
    list_recall_values.append(recall_val)
    list_auc_values.append(roc_auc_val)
    logger.info('seed {}, best_params: {}'.format(idx,roc_auc_val ))

    # pickle.dump(clf_model, open(str(Path.joinpath(consts.PATH_PROJECT_MODELS,
    #                                               'model_clf_{}.sav'.format(generic_name_partition))), 'wb'))

mean_std_specificity = np.mean(list_specificity_values), np.std(list_specificity_values)
mean_std_accuracy = np.mean(list_acc_values), np.std(list_acc_values)
mean_std_recall = np.mean(list_recall_values), np.std(list_recall_values)
mean_std_auc = np.mean(list_auc_values), np.std(list_auc_values)

print('accuracy:', mean_std_accuracy)
print('specificity:', mean_std_specificity)
print('recall:', mean_std_recall)
print('AUC:', mean_std_auc)

exp_name = '{}+{}+{}'.format(args.dataset, args.classifier, args.n_seeds)
# exp_name = '{}+{}'.format(exp_name, 'fs') if args.flag_fs else exp_name

new_row_auc = {'model': exp_name,
               'eval_metric': 'auc',
               'type_encoding': args.type_encoding,
               'mean': mean_std_auc[0],
               'std': mean_std_auc[1]}

new_row_sensitivity = {'model': exp_name,
                       'eval_metric': 'sensitivity',
                       'type_encoding': args.type_encoding,
                       'mean': mean_std_recall[0],
                       'std': mean_std_recall[1]}
new_row_specificity = {'model': exp_name,
                       'eval_metric': 'specificity',
                       'type_encoding': args.type_encoding,
                       'mean': mean_std_specificity[0],
                       'std': mean_std_specificity[1]}

new_row_accuracy = {'model': exp_name,
                    'eval_metric': 'accuracy',
                    'type_encoding': args.type_encoding,
                    'mean': mean_std_accuracy[0],
                    'std': mean_std_accuracy[1]
                    }

if args.augmented:
    if args.with_noise:
        csv_file_path = str(Path.joinpath(consts.PATH_PROJECT_METRICS,
                                          'metrics_classification_{}_{}_aug_bal.csv'.format(args.dataset, args.noise_type)))
    else:
        csv_file_path = str(Path.joinpath(consts.PATH_PROJECT_METRICS,
                                          'metrics_classification_{}_aug_bal.csv'.format(args.dataset)))
else:
    csv_file_path = str(Path.joinpath(consts.PATH_PROJECT_METRICS,
                                      'metrics_classification_{}.csv'.format(args.dataset)))

if os.path.exists(csv_file_path):
    try:
        df_metrics_classification = pd.read_csv(csv_file_path)
    except pd.errors.EmptyDataError:
        df_metrics_classification = pd.DataFrame()
else:
    df_metrics_classification = pd.DataFrame()

df_metrics_classification = pd.concat([df_metrics_classification, pd.DataFrame([new_row_auc])],
                                      ignore_index=True)
df_metrics_classification = pd.concat([df_metrics_classification, pd.DataFrame([new_row_sensitivity])],
                                      ignore_index=True)
df_metrics_classification = pd.concat([df_metrics_classification, pd.DataFrame([new_row_specificity])],
                                      ignore_index=True)
df_metrics_classification = pd.concat([df_metrics_classification, pd.DataFrame([new_row_accuracy])],
                                      ignore_index=True)

# df_metrics_classification = df_metrics_classification.append(new_row_auc, ignore_index=True)
# df_metrics_classification = df_metrics_classification.append(new_row_sensitivity, ignore_index=True)
df_metrics_classification.to_csv(csv_file_path, index=False)


# metrics = mean_std_accuracy, mean_std_recall, mean_std_specificity, mean_std_auc
# current_time = datetime.now()
# str_date_time = current_time.strftime("%m/%d/%Y, %H:%M:%S")
# df_metrics = pd.DataFrame(metrics, columns=['mean', 'std'], index=['accuracy', 'recall', 'specificity', 'auc'])
# df_metrics.to_csv(str(Path.joinpath(consts.PATH_PROJECT_METRICS, args.oversampler,
#                                     'metrics_{}.csv'.format(generic_name))))
