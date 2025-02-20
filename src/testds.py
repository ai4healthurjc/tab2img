import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
import warnings
from sklearn.preprocessing import RobustScaler
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/test/bbdd_fram.csv")
print("Shape of Dataset:", df.shape)

numeric_var = ["age", "cigs_perday", "tot_chol", "sys_bp","dia_bp","bmi","heart_rate","glucose"]
categoric_var = ["gender", "current_smoker", "bp_meds", "prevalent_stroke", "prevalent_hyp", "diabetes", "ed_1", "ed_2", "ed_3","ed_4","target"]


# Cubing transformation
# df['dia_bp_winsorize_cubed'] = df['dia_bp_winsorize'] ** 3
# Reflecting and then applying a log transformation
# Make sure there are no non-positive values in the column before applying a log transformation.
# df['dia_bp_winsorize_reflected'] = -df['dia_bp_winsorize']
# df['dia_bp_winsorize_reflected_log'] = np.log(df['dia_bp_winsorize_reflected'])

# If you want to reflect the log-transformed variable back to its original direction
# df['dia_bp_winsorize_reflected_log'] = -df['dia_bp_winsorize_reflected_log']
# df[["dia_bp_winsorize_reflected", "dia_bp_winsorize_cubed", "dia_bp_winsorize_reflected_log"]].agg(["skew"]).transpose()
# df.drop(["sys_bp", "dia_bp", "glucose", "dia_bp_winsorize", "dia_bp_winsorize_log", "dia_bp_winsorize_sqrt", "dia_bp_winsorize_reflected", "dia_bp_winsorize_reflected_log"], axis=1, inplace=True)
# df_copy = df.copy()
# categoric_var.remove("ed_1")
# categoric_var.remove("ed_2")
# categoric_var.remove("ed_3")
# categoric_var.remove("ed_4")
# df_copy = pd.get_dummies(df_copy, columns = categoric_var[:-1], drop_first = True)

# new_numeric_var = ["age", "sys_bp_winsorize ", "glucose_winsorized", "dia_bp_winsorize_cubed"]

# robus_scaler = RobustScaler()
# Remove any leading/trailing whitespace characters including tabs
# new_numeric_var = [col.strip() for col in new_numeric_var]

# Now apply the robust scaler transformation
# from sklearn.preprocessing import RobustScaler
# robust_scaler = RobustScaler()

# Check if the column names exist in the DataFrame before transformation
# missing_cols = [col for col in new_numeric_var if col not in df_copy.columns]
# if missing_cols:
#     print(f"These columns are missing in the DataFrame: {missing_cols}")
# else:
#     df_copy[new_numeric_var] = robust_scaler.fit_transform(df_copy[new_numeric_var])


# from sklearn.model_selection import train_test_split

# X = df_copy.drop(["target"], axis = 1)
# y = df_copy[["target"]]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 3)
# print(f"X_train: {X_train.shape[0]}")
# print(f"X_test: {X_test.shape[0]}")
# print(f"y_train: {y_train.shape[0]}")
# print(f"y_test: {y_test.shape[0]}")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from deslib.des import DESKNN

numpy = "1.21.5"
pandas = "^1.3.1"


df = pd.read_csv('data/test/bbdd_fram.csv')

# Split data into features and target
X = df.drop('label', axis=1)
y = df['label']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train base classifiers
model_rf = RandomForestClassifier(n_estimators=50, random_state=42)
model_gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
model_lr = LogisticRegression(random_state=42)
model_knn = KNeighborsClassifier(n_neighbors=5)

models = [model_rf, model_gb, model_lr, model_knn]
for model in models:
    model.fit(X_train_scaled, y_train)

# Initialize DES-KNN
desknn = DESKNN(pool_classifiers=models)

# Fit DES-KNN
desknn.fit(X_train_scaled, y_train)

# Make predictions and evaluate
y_pred = desknn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy with DES-KNN: {accuracy}')

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.cm import get_cmap
import numpy as np

# Importing DES techniques from DESlib
from deslib.des import KNORAU, KNORAE, DESP, METADES, KNOP
from deslib.des.des_knn import DESKNN

from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

list_seeds = [1922, 2124, 9832, 2485, 9522]

for seed_value in list_seeds:

    rng = np.random.RandomState(seed_value)

    # Generate a classification dataset
    X, y = make_classification(n_samples=2000, n_classes=3, n_informative=6, random_state=rng)

    # Split the data into training, test, and DSEL (Dynamic Selection Dataset) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=rng)
    X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train, test_size=0.50, random_state=rng)

    # Initialize the pool of classifiers
    pool_classifiers = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=rng)
    pool_classifiers.fit(X_train, y_train)

    # Initialize DES techniques
    knorau = KNORAU(pool_classifiers, random_state=rng)
    kne = KNORAE(pool_classifiers, random_state=rng)
    desp = DESP(pool_classifiers, random_state=rng)
    desknn = DESKNN(pool_classifiers, random_state=rng)
    knop = KNOP(pool_classifiers, random_state=rng)
    meta = METADES(pool_classifiers, random_state=rng)

    # List of methods for iteration
    names = ['KNORA-U', 'KNORA-E', 'DES-P', 'DES-KNN', 'KNOP', 'META-DES']
    methods = [knorau, kne, desp, desknn, knop, meta]

    dict_results = {
        'KNORA-U': [],
        'KNORA-E': [],
        'DES-P': [],
        'DES-KNN': [],
        'KNOP': [],
        'META-DES': []
    }

    # Fit the DES techniques and calculate scores
    scores = []
    for method, name in zip(methods, names):
        method.fit(X_dsel, y_dsel)
        score = method.score(X_test, y_test)
        scores.append(score)
        print("Classification accuracy {} = {:.2f}".format(name, score))
        dict_results[name] += score

    print(dict_results)

    m_val, std_val = np.mean(list_acc_knora_u), np.std(list_acc_knora_u)
    m_val, std_val = np.mean(list_acc_knora_u), np.std(list_acc_knora_u)
    m_val, std_val = np.mean(list_acc_knora_u), np.std(list_acc_knora_u)
    m_val, std_val = np.mean(list_acc_knora_u), np.std(list_acc_knora_u)