import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import utils.consts as consts
from deslib.des.knora_e import KNORAE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from deslib.dcs.a_priori import APriori
from deslib.dcs.mcb import MCB
from deslib.dcs.ola import OLA
from deslib.des.des_p import DESP
from deslib.des.knora_e import KNORAE
from deslib.des.knora_u import KNORAU
from deslib.des.meta_des import METADES


df = pd.read_csv('data/raw/bbdd_fram.csv')

print(df)
print(df.shape)

x_features = df.iloc[:, :-1].values
y_label = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(x_features, y_label, test_size=0.2, random_state=32)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Split the data into training and DSEL for DS techniques
X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train, test_size=0.5, random_state=32)

pool_classifiers = RandomForestClassifier(n_estimators=20)
pool_classifiers.fit(X_train, y_train)

# Initialize the DS techniques
knorau = KNORAU(pool_classifiers, k=15)
kne = KNORAE(pool_classifiers)
desp = DESP(pool_classifiers)
ola = OLA(pool_classifiers)
mcb = MCB(pool_classifiers, random_state=32)

knorau.fit(X_dsel, y_dsel)
kne.fit(X_dsel, y_dsel)
desp.fit(X_dsel, y_dsel)
ola.fit(X_dsel, y_dsel)
mcb.fit(X_dsel, y_dsel)

###############################################################################
# Evaluating the methods
# -----------------------
# Let's now evaluate the methods on the test set. We also use the performance
# of Bagging (pool of classifiers without any selection) as a baseline
# comparison. We can see that  the majority of DS methods achieve higher
# classification accuracy.

print('Evaluating DS techniques:')
print('Classification accuracy KNORA-Union: ', knorau.score(X_test, y_test))
print('Classification accuracy KNORA-Eliminate: ', kne.score(X_test, y_test))
print('Classification accuracy DESP: ', desp.score(X_test, y_test))
print('Classification accuracy OLA: ', ola.score(X_test, y_test))
print('Classification accuracy MCB: ', mcb.score(X_test, y_test))
print('Classification accuracy Bagging: ', pool_classifiers.score(X_test, y_test))

