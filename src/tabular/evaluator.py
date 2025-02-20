import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from typing import Tuple
import warnings
from sklearn.metrics._classification import _check_targets, _prf_divide, precision_recall_fscore_support, unique_labels
from sklearn.preprocessing import LabelEncoder
# from utils.plotter import plot_confusion_matrix
from sklearn.model_selection import train_test_split


list_multiclass_avg = ['micro', 'macro', 'weighted']


def get_metric_classification(scoring_metric, n_classes, average='macro'):

    if n_classes == 2:
        return scoring_metric
    else: # multiclass
        if scoring_metric == 'roc_auc':
            return '{}_{}'.format(scoring_metric, 'ovo')
        elif scoring_metric == 'f1':
            return '{}_{}'.format(scoring_metric, average)


def compute_classification_prestations(y_true: np.array,
                                       y_pred: np.array,
                                       class_names: np.array,
                                       average='micro',
                                       verbose=False,
                                       save_confusion_matrix=False
                                       ) -> (float, float, float, float):

    if len(np.unique(y_pred)) != len(class_names):
        y_pred = np.where(y_pred >= 0.5, 1.0, y_pred)
        y_pred = np.where(y_pred < 0.5, 0.0, y_pred)

    if verbose:
        print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)

    if save_confusion_matrix:
        plot_confusion_matrix(cm, class_names, save_confusion_matrix)

    n_classes = len(set(y_true))

    if n_classes == 2:

        return {'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'specificity': specificity_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'roc_auc': roc_auc_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred)
        }

    else:
        return compute_multiclass_metrics(y_true, y_pred, average)

    # fpr, tpr, threshold = roc_curve(y_true, y_pred)
    # roc_auc = auc(fpr, tpr)
    # plot_auroc(fpr, tpr, roc_auc)


def compute_multiclass_metrics(y_true, y_pred, avg='micro', as_frame=False):

    dict_metrics_report = {'precision': precision_score(y_true, y_pred, average=avg),
                           'specificity': specificity_score(y_true, y_pred, average=avg),
                           'recall': recall_score(y_true, y_pred, average=avg),
                           'roc_auc': roc_auc_score(y_true, y_pred, multi_class='ovo'),
                           'f1': f1_score(y_true, y_pred, average=avg)
    }

    # if as_frame:
    #     return pd.DataFrame(dict_metrics_report)

    return dict_metrics_report


def specificity_score(y_true,
                      y_pred,
                      labels=None,
                      pos_label=1,
                      average='binary',
                      sample_weight=None):
    """Compute the specificity

    The specificity is the ratio ``tp / (tp + fn)`` where ``tp`` is the number
    of true positives and ``fn`` the number of false negatives. The specificity
    is intuitively the ability of the classifier to find all the positive
    samples.

    The best value is 1 and the worst value is 0.

    Parameters
    ----------
    y_true : ndarray, shape (n_samples, )
        Ground truth (correct) target values.

    y_pred : ndarray, shape (n_samples, )
        Estimated targets as returned by a classifier.

    labels : list, optional
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average.

    pos_label : str or int, optional (default=1)
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : str or None, optional (default=None)
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    warn_for : tuple or set, for internal use
        This determines which warnings will be made in the case that this
        function is being used to return only one of its metrics.

    sample_weight : ndarray, shape (n_samples, )
        Sample weights.

    Examples
    --------
    >>> import numpy as np
    >>> from imblearn.metrics import specificity_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> specificity_score(y_true, y_pred, average='macro')
    0.66666666666666663
    >>> specificity_score(y_true, y_pred, average='micro')
    0.66666666666666663
    >>> specificity_score(y_true, y_pred, average='weighted')
    0.66666666666666663
    >>> specificity_score(y_true, y_pred, average=None)
    array([ 0.75,  0.5 ,  0.75])

    Returns
    -------
    specificity : float (if ``average`` = None) or ndarray, \
        shape (n_unique_labels, )

    """
    _, s, _ = sensitivity_specificity_support(
        y_true,
        y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        warn_for=('specificity', ),
        sample_weight=sample_weight)

    return s


def sensitivity_specificity_support(y_true,
                                    y_pred,
                                    labels=None,
                                    pos_label=1,
                                    average=None,
                                    warn_for=('sensitivity', 'specificity'),
                                    sample_weight=None):

    average_options = (None, 'micro', 'macro', 'weighted', 'samples')
    if average not in average_options and average != 'binary':
        raise ValueError('average has to be one of ' + str(average_options))

    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    present_labels = unique_labels(y_true, y_pred)

    if average == 'binary':
        if y_type == 'binary':
            if pos_label not in present_labels:
                if len(present_labels) < 2:
                    return (0., 0., 0)
                else:
                    raise ValueError("pos_label=%r is not a valid label: %r" %
                                     (pos_label, present_labels))
            labels = [pos_label]
        else:
            raise ValueError("Target is %s but average='binary'. Please "
                             "choose another average setting." % y_type)
    elif pos_label not in (None, 1):
        warnings.warn("Note that pos_label (set to %r) is ignored when "
                      "average != 'binary' (got %r). You may use "
                      "labels=[pos_label] to specify a single positive class."
                      % (pos_label, average), UserWarning)

    if labels is None:
        labels = present_labels
        n_labels = None
    else:
        n_labels = len(labels)
        labels = np.hstack(
            [labels, np.setdiff1d(
                present_labels, labels, assume_unique=True)])

    if y_type.startswith('multilabel'):
        raise ValueError('imblearn does not support multilabel')
    elif average == 'samples':
        raise ValueError("Sample-based precision, recall, fscore is "
                         "not meaningful outside multilabel "
                         "classification. See the accuracy_score instead.")
    else:
        le = LabelEncoder()
        le.fit(labels)
        y_true = le.transform(y_true)
        y_pred = le.transform(y_pred)
        sorted_labels = le.classes_

        # labels are now from 0 to len(labels) - 1 -> use bincount
        tp = y_true == y_pred
        tp_bins = y_true[tp]
        if sample_weight is not None:
            tp_bins_weights = np.asarray(sample_weight)[tp]
        else:
            tp_bins_weights = None

        if len(tp_bins):
            tp_sum = np.bincount(
                tp_bins, weights=tp_bins_weights, minlength=len(labels))
        else:
            true_sum = pred_sum = tp_sum = np.zeros(len(labels))
        if len(y_pred):
            pred_sum = np.bincount(
                y_pred, weights=sample_weight, minlength=len(labels))
        if len(y_true):
            true_sum = np.bincount(
                y_true, weights=sample_weight, minlength=len(labels))

        # Compute the true negative
        tn_sum = y_true.size - (pred_sum + true_sum - tp_sum)

        # Retain only selected labels
        indices = np.searchsorted(sorted_labels, labels[:n_labels])
        tp_sum = tp_sum[indices]
        true_sum = true_sum[indices]
        pred_sum = pred_sum[indices]
        tn_sum = tn_sum[indices]

    if average == 'micro':
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])
        tn_sum = np.array([tn_sum.sum()])

    with np.errstate(divide='ignore', invalid='ignore'):
        specificity = _prf_divide(tn_sum, tn_sum + pred_sum - tp_sum,
                                  'specificity', 'predicted', average,
                                  warn_for)
        sensitivity = _prf_divide(tp_sum, true_sum, 'sensitivity', 'true',
                                  average, warn_for)

    if average == 'weighted':
        weights = true_sum
        if weights.sum() == 0:
            return 0, 0, None
    elif average == 'samples':
        weights = sample_weight
    else:
        weights = None

    if average is not None:
        assert average != 'binary' or len(specificity) == 1
        specificity = np.average(specificity, weights=weights)
        sensitivity = np.average(sensitivity, weights=weights)
        true_sum = None  # return no support

    return sensitivity, specificity, true_sum


def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, np.array]:
    """Split the data in a stratified way.

    Returns:
        A tuple containing train dataset, test data and test label.
    """

    train_data, test_data = train_test_split(data, stratify=data[[LABEL]], random_state=1113
    )
    _train_ds = ray.data.from_pandas(train_data)
    _test_label = test_data[LABEL].values
    _test_df = test_data.drop([LABEL], axis=1)
    return _train_ds, _test_df, _test_label


def evaluate_classification(clf_model, x_features, y_label):

    kfold = model_selection.KFold(n_splits=5)

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, pos_label=1),
        'recall': make_scorer(recall_score, pos_label=1),
        'f1_score': make_scorer(f1_score, pos_label=1)
    }

    results = model_selection.cross_validate(clf_model, x_features, y_label, cv=kfold, scoring=scoring)

    print("Accuracy", np.mean(results['test_accuracy']))
    print("Precision", np.mean(results['test_precision']))
    print("Recall", np.mean(results['test_recall']))
    print("F1 Score", np.mean(results['test_f1_score']))