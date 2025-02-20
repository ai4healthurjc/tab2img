import numpy as np
from deslib.dcs import OLA, MCB
from deslib.des import DESP, KNORAU, KNOP, METADES
from deslib.des.knora_e import KNORAE
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from deslib.static import StackedClassifier, SingleBest, StaticSelection

import utils.consts as consts


def check_static_estimators(pool_classifiers):
    # Setting up static methods.
    stacked = StackedClassifier(pool_classifiers)
    static_selection = StaticSelection(pool_classifiers)
    single_best = SingleBest(pool_classifiers)


def train_des_estimator(X_train, y_train, model_name='meta-des', n_estimators=10, k=5, random_state=1234):
    """
    Train a specific DES classifier

    Each selected classifier has a number of votes equals to the number of samples
    in the region of competence that it predicts the correct label. The votes
    obtained by all base classifiers are aggregated to obtain the final
    ensemble decision.

    Parameters
    ----------
    X_train: np.array
        numpy array for training models
    y_train: np.array
        numpy array that identifies label in the classification
    model_name: string (Default = 'meta-des')
        name of the DES classifier
    n_estimators: int (Default = 10)
        number of estimators used in the ensemble
    k : int (Default = 5)
        Number of neighbors used to estimate the competence of the base classifiers.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    rng = np.random.RandomState(random_state)
    X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train, test_size=0.50, random_state=rng)

    # As base estimator DT is selected because it works reasonably well in mixed-type data
    pool_classifiers = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                         n_estimators=n_estimators,
                                         random_state=rng
                                         )

    pool_classifiers.fit(X_train, y_train)

    list_des_estimators = consts.LIST_DYNAMIC_CLASSIFIER_SELECTION_ESTIMATORS



    # Initialize a DS technique. Here we specify the size of
    # the region of competence (5 neighbors)
    knorau = KNORAU(pool_classifiers, k=k, random_state=rng)
    kne = KNORAE(pool_classifiers, k=k, random_state=rng)
    desp = DESP(pool_classifiers, k=k, random_state=rng)
    ola = OLA(pool_classifiers, k=k, random_state=rng)
    mcb = MCB(pool_classifiers, k=k, random_state=rng)
    knop = KNOP(pool_classifiers, k=k, random_state=rng)
    meta = METADES(pool_classifiers, k=k, random_state=rng)

    names = ['Single Best', 'Static Selection', 'Stacked',
             'KNORA-U', 'KNORA-E', 'DES-P', 'OLA', 'MCB', 'KNOP', 'META-DES']

    methods = [single_best, static_selection, stacked,
               knorau, kne, desp, ola, mcb, knop, meta]

    scores = []
    for method, name in zip(methods, names):
        method.fit(X_dsel, y_dsel)
        scores.append(method.score(X_test, y_test))
        print("Classification accuracy {} = {}".format(name, method.score(X_test, y_test)))
