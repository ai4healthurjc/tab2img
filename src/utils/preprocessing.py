from utils.loader import load_by_dataset_name
import numpy as np
import pandas as pd
import category_encoders as ce
import math
import logging
import coloredlogs
from gower import gower_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

def preprocess_for_clustering(dataset):
    dataset = pd.DataFrame(dataset)
    dataset.columns = [str(e) for e in range(0, len(dataset.iloc[0]))]
    dataset, categorical_columns, numeric_columns = identify_feature_type(dataset, categorical_threshold=0.05,
                                                                          impute_missing=False)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(dataset)
    cat = [False] * len(dataset.iloc[0])
    for e in categorical_columns:
        cat[int(e)] = True
    dist_matrix = gower_matrix(data_scaled, cat_features=cat)
    return dist_matrix, data_scaled


def preprocess_for_fs(dataset,label,seed):
    dataset = pd.DataFrame(dataset)

    dataset, categorical_columns, numeric_columns = identify_feature_type(dataset, categorical_threshold=0.05,
                                                                          impute_missing=False)
    X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=seed)
    col = X_train.columns
    scaler = MinMaxScaler()
    data_train_scaled = scaler.fit_transform(X_train)
    data_train_scaled= pd.DataFrame(data_train_scaled,columns=col)
    return data_train_scaled, y_train, categorical_columns,numeric_columns


def count_null_values(df_features, null_value='?'):
    # path_dataset, features, y_label = load_by_dataset_name(dataset_name)
    features = df_features.replace(null_value, np.nan)
    null_values = features.isnull().sum()
    print(null_values)


def remove_null_variables(df_features, y_label, threshold=0.5, null_value="?"):
    count_null_values(df_features)
    # Replace "?" with NaN to enable the detection of null values
    features = df_features.replace(null_value, np.nan)
    # Calculate the proportion of null values per variable
    null_proportions = features.isnull().mean()

    # Select variables with more than the threshold of null values
    null_variables = null_proportions[null_proportions > threshold].index

    # Remove the null variables from the DataFrame
    df_without_nulls = features.drop(null_variables, axis=1)

    print("The variables with more than 60% of the values as null are:", null_variables)
    print(df_without_nulls)
    print(y_label.value_counts())
 
    return df_without_nulls, y_label
def remove_unique_variables(dataset_preprocess):
    num_unique_values = dataset_preprocess.nunique()
    single_valued_columns = num_unique_values[num_unique_values == 1].index
    df = dataset_preprocess.drop(columns=single_valued_columns)
    return df


def identify_feature_type(dataset, categorical_threshold=0.05, impute_missing=True):
    categorical_columns = []
    numeric_columns = []

    for col in dataset.columns:
        unique_values = dataset[col].nunique()
        total_values = dataset.shape[0]
        uniqueness_ratio = unique_values / total_values

        if uniqueness_ratio <= categorical_threshold:
            categorical_columns.append(col)
        else:
            numeric_columns.append(col)

    dataset[numeric_columns] = dataset[numeric_columns].apply(pd.to_numeric, errors='coerce')

    if impute_missing:
        mode_imputer = dataset[categorical_columns].mode().iloc[0]
        dataset[categorical_columns] = dataset[categorical_columns].fillna(mode_imputer)
        mean_imputer = dataset[numeric_columns].mean()
        dataset[numeric_columns] = dataset[numeric_columns].fillna(mean_imputer)
        dataset = dataset.dropna(axis=1, how='any')

        for col in categorical_columns:
            if col in ['ca', 'tal']:
                dataset[col] = dataset[col].astype(float).astype(int)
            if dataset[col].dtype == 'O' and dataset[col].str.isnumeric().all():
                dataset[col] = dataset[col].astype(int)

        unique_counts = dataset.nunique()

        # Find columns with only a single unique value
        columns_to_drop = unique_counts[unique_counts == 1].index
        print(columns_to_drop)
        # Remove columns
        dataset = dataset.drop(columns=columns_to_drop)

        # Remove columns with a single unique value from lists
        for col in columns_to_drop:
            if col in numeric_columns:
                numeric_columns.remove(col)
            if col in categorical_columns:
                categorical_columns.remove(col)

    return dataset, categorical_columns, numeric_columns


def preprocess_categorical_features(dataset):
    dataset, categorical_columns, numeric_columns = identify_feature_type(dataset, categorical_threshold=0.05)
    for col in categorical_columns:
        print(col)
        print(dataset[col].unique())
        if set(dataset[col].unique()) == {1, 2}:
            # Perform replacement if both values are present
            dataset[col] = dataset[col].replace(2, 0)
        if set(dataset[col].unique()) == {'g', 'b'}:
            # Perform replacement if both values are present
            dataset[col] = dataset[col].replace('g', 0)
            dataset[col] = dataset[col].replace('b', 1)
        if set(dataset[col].unique()) == {'b', 'a'}:
            # Perform replacement if both values are present
            dataset[col] = dataset[col].replace('b', 1)
            dataset[col] = dataset[col].replace('a', 0)
        if set(dataset[col].unique()) == {'t', 'f'}:
            # Perform replacement if both values are present
            dataset[col] = dataset[col].replace('t', 1)
            dataset[col] = dataset[col].replace('f', 0)
        if set(dataset[col].unique()) == {"b'2'", "b'1'"}:
            # Perform replacement if both values are present
            dataset[col] = dataset[col].replace("b'1'", 1).replace("b'2'", 0)
        # USE THESE COMMENTS FOR CMC
        # if col!="Husband's occupation":
        #     if set(dataset[col].unique()) == {1, 2, 3, 4}:
        #         #print(col)
        #         # Perform replacement if both values are present
        #         dataset[col] = dataset[col].replace(2, 1)
        #         dataset[col] = dataset[col].replace(3, 1)
        #         dataset[col] = dataset[col].replace(4, 1)
        #         dataset[col] = dataset[col].replace(1, 0)

        if col == 'slope':
            dataset[col] = dataset[col].replace(1, 'Unsloping')
            dataset[col] = dataset[col].replace(2, 'Flat')
            dataset[col] = dataset[col].replace(3, 'Dowsloping')
        if col == 'cp':
            dataset[col] = dataset[col].replace(1, 'typical angina')
            dataset[col] = dataset[col].replace(2, 'atypical angina')
            dataset[col] = dataset[col].replace(3, 'non_anginal pain')
            dataset[col] = dataset[col].replace(4, 'asymptomatic')
        if col == 'tal':
            dataset[col] = dataset[col].replace(3, 0)
            dataset[col] = dataset[col].replace(7, 1)
            dataset[col] = dataset[col].replace(6, 2)

        if col == 'CLEAR-G':
            dataset[col] = dataset[col].replace('N', 0)
            dataset[col] = dataset[col].replace('G', 1)
        if col == 'T-OR-D':
            dataset[col] = dataset[col].replace('THROUGH', 0)
            dataset[col] = dataset[col].replace('DECK', 1)
        # TAE
    # for col2 in numeric_columns:
    #     if col2 == 'Course instructor':
    #         dataset = pd.get_dummies(dataset, columns=[col2])
    #     if col2 == 'Course':
    #         dataset = pd.get_dummies(dataset, columns=[col2])

    object_categorical_columns = [col for col in categorical_columns if dataset[col].dtype == 'object']
    print(object_categorical_columns)

    if object_categorical_columns:
        dataset = pd.get_dummies(dataset, columns=object_categorical_columns)
    print(dataset)
    print(dataset.dtypes)

    return dataset


def preprocess_categorical_features_different(dataset, encoding_type, impute_missing=True):

    _, categorical_columns, numeric_columns = identify_type_features(dataset, discrete_threshold=10)

    # dataset, categorical_columns, numeric_columns = identify_feature_type(dataset, categorical_threshold=0.05)
    categorical_columns = list(categorical_columns)
    numeric_columns = list(numeric_columns)

    if impute_missing:
        dataset, categorical_columns, numeric_columns = impute_null_values(dataset, categorical_columns,
                                                                           numeric_columns)

    if encoding_type == 'count':
        encoder = ce.CountEncoder(cols=categorical_columns)
    elif encoding_type == 'binary':
        encoder = ce.BinaryEncoder(cols=categorical_columns)
    elif encoding_type == 'one_hot':
        encoder = ce.OneHotEncoder(cols=categorical_columns)
    elif encoding_type == 'rank_hot':
        encoder = ce.RankHotEncoder(cols=categorical_columns)
    elif encoding_type == 'helmert':
        encoder = ce.HelmertEncoder(cols=categorical_columns, drop_invariant=True)
    elif encoding_type == 'gray':
        encoder = ce.GrayEncoder(cols=categorical_columns)
    elif encoding_type == 'basen':
        encoder = ce.BaseNEncoder(cols=categorical_columns)
    elif encoding_type == 'polynomial':
        encoder = ce.PolynomialEncoder(cols=categorical_columns)

    data_encoded = encoder.fit_transform(dataset)

    return data_encoded

# def preprocess_categorical_features_binary(dataset):
#     dataset, categorical_columns, numeric_columns=identify_feature_type(dataset, categorical_threshold=0.05)
    
#     encoder = ce.CountEncoder(cols=categorical_columns)
#     data_encoded = encoder.fit_transform(dataset)

#     # Now 'data_encoded' contains the variables encoded with Count Encoding only for the specified columns.
#     print(data_encoded)

#     return data_encoded


def save_dataset(dataset_name, dataset, label, path_save):
    dataset_con_label = pd.concat([dataset, label], axis=1)
    np.unique(label)
    logger.info('dataset: {}, #samples: {}, #features: {}'.format(dataset_name, dataset_con_label.shape[0],
                                                                  dataset_con_label.shape[1]))
    dataset_con_label.to_csv(path_save, index=False)


def identify_binary_and_numeric_features(dataset):
    binary_variables = []
    numeric_variables = []

    for columns in dataset.columns:
        # If the variable has only 2 unique values, it is considered binary
        if dataset[columns].nunique() == 2:
            binary_variables.append(columns)
        # If the variable is of numeric type, it is considered numeric
        elif pd.api.types.is_numeric_dtype(dataset[columns]):
            numeric_variables.append(columns)

    return binary_variables, numeric_variables


def identify_type_features(df, discrete_threshold=10, debug=False):
    """
    Categorize every feature/column of df as discrete or continuous according to whether or not the unique responses
    are numeric and, if they are numeric, whether there are fewer unique responses than discrete_threshold.
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

        if len(responses) == 2:
            type_var = 'binary'
        elif len(responses) < discrete_threshold or string_count > 0:
            type_var = 'categorical'
        else:
            type_var = 'numerical'

        types.append(type_var)
        # types.append('d' if len(responses) < discrete_threshold or string_count > 0 else 'c')
        # types.append('d' if len(responses) < discrete_threshold or string_count > 0 else 'c')

    feature_info = pd.DataFrame({'count': counts,
                                 'string_count': string_counts,
                                 'float_count': float_counts,
                                 'null_count': null_counts,
                                 'type': types}, index=df.columns)
    if debug:
        print(f'Counted {sum(feature_info["type"] == "d")} discrete features and {sum(feature_info["type"] == "c")} continuous features')

    v_bin_vars = feature_info[feature_info['type'] == 'binary'].index
    v_cat_vars = feature_info[feature_info['type'] == 'categorical'].index
    numerical_vars = feature_info[feature_info['type'] == 'numerical'].index
    categorical_vars = np.concatenate((v_bin_vars, v_cat_vars))

    return feature_info, categorical_vars, numerical_vars


def impute_null_values(dataset, categorical_columns, numeric_columns):
    dataset[numeric_columns] = dataset[numeric_columns].apply(pd.to_numeric, errors='coerce')
    mode_imputer = dataset[categorical_columns].mode().iloc[0]
    dataset[categorical_columns] = dataset[categorical_columns].fillna(mode_imputer)
    mean_imputer = dataset[numeric_columns].mean()
    dataset[numeric_columns] = dataset[numeric_columns].fillna(mean_imputer)
    dataset = dataset.dropna(axis=1, how='any')

    for col in categorical_columns:
        if col in ['ca', 'tal']:
            dataset[col] = dataset[col].astype(float).astype(int)
        if dataset[col].dtype == 'O' and dataset[col].str.isnumeric().all():
            dataset[col] = dataset[col].astype(int)

    unique_counts = dataset.nunique()

    # Find columns with only a single unique value
    columns_to_drop = unique_counts[unique_counts == 1].index
    print(columns_to_drop)
    # Remove columns
    dataset = dataset.drop(columns=columns_to_drop)

    # Remove columns with a single unique value from lists
    for col in columns_to_drop:
        if col in numeric_columns:
            numeric_columns.remove(col)
        if col in categorical_columns:
            categorical_columns.remove(col)

    return dataset, categorical_columns, numeric_columns
