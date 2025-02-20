import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import utils.consts as consts
import logging
import coloredlogs
import glob


logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def load_by_dataset_name(dataset_id, type_dataset):

    if dataset_id in consts.DICT_DATASETS_MAP:
        dataset_name = consts.DICT_DATASETS_MAP.get(dataset_id)
    else:
        raise ValueError('Dataset not found!')

    path_dataset = Path.joinpath(consts.PATH_PROJECT_DATA_RAW,
                                 'bbdd_{}'.format(type_dataset), '{}.csv'.format(dataset_name))

    if dataset_id == 'spambase':
        df = pd.read_csv(str(path_dataset))
        y_label = df.iloc[:, -1]
        features = df.iloc[:, :-1]

    if dataset_id == 'crx':
        # path_dataset = Path(consts.PATH_PROJECT_DATA_CRX_DIR)
        df = pd.read_csv(str(path_dataset))
        y_label = df['label']
        if set(y_label.unique()) == {'+', '-'}:
            y_label = y_label.replace('+', 0).replace('-', 1)
        features = df.drop('label', axis=1)

    if dataset_id == 'diabetes':
        path_dataset = Path(consts.PATH_PROJECT_DATA_DIABETES_DIR)
        print(path_dataset)
        df = pd.read_csv(str(path_dataset))
        y_label = df['label']
        if set(y_label.unique()) == {"Positive", "Negative"}:
            # Perform replacement if both values are present
            y_label = y_label.replace("Positive", 1).replace("Negative", 0)

        features = df.drop('label', axis=1)
    
    if dataset_id == 'german':
        path_dataset = Path(consts.PATH_PROJECT_DATA_GERMAN_DIR)
        df = pd.read_csv(str(path_dataset), delimiter=' ')
        y_label = df['label']
        y_label = y_label-1

        features = df.drop('label', axis=1)

    if dataset_id == 'hepatitis':
        path_dataset = Path(consts.PATH_PROJECT_DATA_HEPATITIS_DIR)
        df = pd.read_csv(str(path_dataset))
        y_label = df['label']

        if set(y_label.unique()) == {1, 2}:
            # Perform replacement if both values are present
            y_label = y_label.replace(2, 0)

        features = df.drop('label', axis=1)

    if dataset_id == 'ionos':
        path_dataset = Path(consts.PATH_PROJECT_DATA_IONOS_DIR)
        print(path_dataset)
        df = pd.read_csv(str(path_dataset))
        print(df)

        y_label = df['label']
        if set(y_label.unique()) == {'b', 'g'}:
            # Perform replacement if both values are present
            y_label = y_label.replace('b', 1).replace('g', 0)

        features = df.drop('label', axis=1)

    if dataset_id == 'saheart':
        path_dataset = Path(consts.PATH_PROJECT_DATA_SAHEART_DIR)
        print(path_dataset)
        # data, meta = arff.loadarff(path_dataset)
        # Converts data to a Pandas DataFrame
        # df = pd.DataFrame(data)
        # df.to_csv('saheart.csv', index=False)

        df = pd.read_csv(str(path_dataset))
        y_label = df['label']
        if set(y_label.unique()) == {"b'2'", "b'1'"}:
            # Perform replacement if both values are present
            y_label = y_label.replace("b'1'", 0).replace("b'2'", 1)

        features = df.drop('label', axis=1)

    if dataset_id == 'australian':
        path_dataset = Path(consts.PATH_PROJECT_DATA_AUSTRALIAN_DIR)
        print(path_dataset)

        df = pd.read_csv(str(path_dataset), delimiter=' ')
        y_label = df['label']
        features = df.drop('label', axis=1)

    if dataset_id == 'horse':
        path_dataset = Path(consts.PATH_PROJECT_DATA_HORSE_DIR)
        print(path_dataset)

        df = pd.read_csv(str(path_dataset), delimiter=' ')
        y_label = df['label']
        
        features = df.drop('label', axis=1)
        features = features.drop('Hospital_Number', axis=1)

    if dataset_id == 'cylinder':
        path_dataset = Path(consts.PATH_PROJECT_DATA_CYLINDER_DIR)
        print(path_dataset)

        df = pd.read_csv(str(path_dataset))
        df = df.dropna(subset=['label'])
        y_label = df['label']
        y_label = y_label.replace('band', 1).replace('noband', 0)
        print(y_label)
        features = df.drop('label', axis=1)
        features = features.drop('timestamp', axis=1)

    if dataset_id == 'dresses':
        path_dataset = Path(consts.PATH_PROJECT_DATA_DRESSES_DIR)
        print(path_dataset)

        df = pd.read_csv(str(path_dataset))
        y_label = df['label']
        y_label = y_label - 1
        features = df.drop('label', axis=1)
        features = features.drop('id', axis=1)

    if dataset_id == 'loan':
        path_dataset = Path(consts.PATH_PROJECT_DATA_LOAN_DIR)
        print(path_dataset)

        df = pd.read_csv(str(path_dataset))
        y_label = df['label']
        y_label = y_label.replace('N', 0).replace('Y', 1)
        features = df.drop('label', axis=1)
        features = features.drop('Loan_ID', axis=1)

    if dataset_id == 'autism':
        path_dataset = Path(consts.PATH_PROJECT_DATA_AUTISM_DIR)
        print(path_dataset)

        df = pd.read_csv(str(path_dataset))
        y_label = df['label']
        y_label = y_label.replace('NO', 0).replace('YES', 1)
        features = df.drop('label', axis=1)
        features = features.drop('id', axis=1)
        features = features.drop('result', axis=1)

    if dataset_id == 'thoracic':
        path_dataset = Path(consts.PATH_PROJECT_DATA_THORACIC_DIR)
        print(path_dataset)

        df = pd.read_csv(str(path_dataset))
        y_label = df['label']
        y_label = y_label.replace('F', 0).replace('T', 1)
        features = df.drop('label', axis=1)
        features = features.drop('id', axis=1)

    if dataset_id == 'dermat':
        path_dataset = Path(consts.PATH_PROJECT_DATA_DERMAT_DIR)
        print(path_dataset)

        df = pd.read_csv(str(path_dataset))
        y_label = df['label']
        y_label = y_label - 1

        features = df.drop('label', axis=1)

    if dataset_id == 'cmc':
        path_dataset = Path(consts.PATH_PROJECT_DATA_CMC_DIR)
        print(path_dataset)

        df = pd.read_csv(str(path_dataset))
        y_label = df['label']
        y_label = y_label - 1

        features = df.drop('label', axis=1)

    if dataset_id == 'heart':
        path_dataset = Path(consts.PATH_PROJECT_DATA_HEART_DIR)
        print(path_dataset)

        df = pd.read_csv(str(path_dataset))
        print(df)
        y_label = df['label']
        features = df.drop('label', axis=1)

    if dataset_id == 'tae':
        path_dataset = Path(consts.PATH_PROJECT_DATA_TAE_DIR)
        print(path_dataset)

        df = pd.read_csv(str(path_dataset))
        y_label = df['label']
        y_label = y_label - 1

        features = df.drop('label', axis=1)

    if dataset_id == 'anneal':
        path_dataset = Path(consts.PATH_PROJECT_DATA_ANNEALING_DIR)
        print(path_dataset)

        df = pd.read_csv(str(path_dataset))
        y_label = df['label']
        y_label = y_label.replace("U", 4)
        y_label = y_label.astype(int)

        y_label = y_label - 1

        features = df.drop('label', axis=1)

    if dataset_id == 'bridges':
        path_dataset = Path(consts.PATH_PROJECT_DATA_BRIDGES_DIR)
        print(path_dataset)

        df = pd.read_csv(str(path_dataset))
        df = df[df['label'] != '?']
        y_label = df['label']
        y_label = y_label.replace("STEEL", 0).replace("WOOD", 1).replace("IRON", 2)
        features = df.drop(['IDENTIF', 'TYPE', 'label'], axis=1)

    if dataset_id == 'postop':
        path_dataset = Path(consts.PATH_PROJECT_DATA_POST_DIR)
        print(path_dataset)

        df = pd.read_csv(str(path_dataset))
        print(df)
        y_label = df['label']
        y_label = y_label.replace("A", 0).replace("S", 1).replace("I", 2)
        features = df.drop(['label'], axis=1)

    if dataset_id == 'hypo':
        path_dataset = Path(consts.PATH_PROJECT_DATA_HYPO_DIR)
        print(path_dataset)

        df = pd.read_csv(str(path_dataset))
        print(df)
        y_label = df['label']
        y_label = y_label.replace("negative", 0).replace("compensated_hypothyroid", 1).replace("primary_hypothyroid", 1).replace("secondary_hypothyroid", 1)
        features = df.drop(['label'], axis=1)
        features = features.drop(['id'], axis=1)

    if dataset_id == 'autos':
        path_dataset = Path(consts.PATH_PROJECT_DATA_AUTOS_DIR)
        print(path_dataset)

        df = pd.read_csv(str(path_dataset))
        print(df)
        y_label = df['label']
        y_label = y_label.replace(-3, 0).replace(-2, 1).replace(-1, 2).replace(0, 3).replace(1, 0).replace(2, 1).replace(3, 2)
        features = df.drop(['label'], axis=1)

    if dataset_id == 'diabetes_num':
        path_dataset = Path(consts.PATH_PROJECT_DATA_DIABETES_NUM_DIR)
        # print(path_dataset)
        df = pd.read_csv(str(path_dataset))
        df['Gender'] = df['Gender'].replace('f', 0).replace('F', 0).replace('M', 1)

        y_label = df['label']
        y_label = y_label.astype(str)

        # print(y_label.value_counts())
        y_label = y_label.replace('P', 1).replace('Y', 1).replace('N', 0).replace('Y ', 1).replace('N ', 0) 
        features = df.drop('label', axis=1)
        features = features.drop('ID', axis=1)
        features = features.drop('No_Pation', axis=1)

        # y_label = y_label.replace("U", 4)
        # y_label=y_label.astype(int)

        # y_label = y_label - 1

    if dataset_id == 'fram':
        path_dataset = Path(consts.PATH_PROJECT_DATA_FRAM_DIR)
        print(path_dataset)

        df = pd.read_csv(str(path_dataset))
        df = df.rename(columns={'gender': 'sex', 'currentSmoker': 'smoker',
                                'sys_bp': 'sbp', 'ten_year_chd': 'label'})
        print(df)
        features_names = ['sex', 'age', 'smoker', 'sbp', 'glucose']
        features = df[features_names]
        y_label = df['label']

    if dataset_id == 'steno':
        path_dataset = Path(consts.PATH_PROJECT_DATA_STENO_DIR)
        print(path_dataset)

        df = pd.read_csv(str(path_dataset))
        df = df.rename(columns={'dm_duration': 'diabetes', 'smoking': 'smoker',
                                'hba1c': 'glucose', 'cvd_risk_10y': 'label'})
        print(df)
        features_names = ['sex', 'age', 'smoker', 'sbp', 'glucose']
        features = df[features_names]
        features.loc[:, 'glucose'] = features['glucose'].apply(lambda x: ((x /10.929 )+2.15)* 28.7 - 46.7)
        y_label = (df['label'] > 0.2).astype(int)

    return path_dataset, features, y_label


def get_image_path_from_id(dataset_id, noise_type):

    if noise_type == 'homogeneous':
        dict_datasets_map = consts.DICT_IMAGES_MAP_HOMO
    if noise_type == 'heterogeneous':
        dict_datasets_map = consts.DICT_IMAGES_MAP_HETE
    if noise_type == 'cluster':
        dict_datasets_map = consts.DICT_IMAGES_MAP_CLUSTER
    if noise_type == 'preprocessed':
        dict_datasets_map = consts.DICT_IMAGES_MAP

    if dataset_id in dict_datasets_map:
        return dict_datasets_map.get(dataset_id)
    else:
        return None


def get_image_path_from_id_interpretability(dataset_id, noise_type):

    if noise_type == 'homogeneous':
        dict_datasets_map = consts.DICT_IMAGES_MAP_HOMO_INTERPRETABILITY
    elif noise_type == 'heterogeneous':
        dict_datasets_map = consts.DICT_IMAGES_MAP_HETE_INTERPRETABILITY
    else:
        dict_datasets_map = consts.DICT_IMAGES_MAP_INTERPRETABILITY

    if dataset_id in dict_datasets_map:
        return dict_datasets_map.get(dataset_id)
    else:
        return None    


def get_dataset_path_from_id(dataset_id, name):
    if name == 'preprocessed':
        dict_datasets_map = consts.DICT_DATASETS_MAP
    elif name == 'homogeneous':
        dict_datasets_map = consts.DICT_DATASETS_MAP_HOMO
    elif name == 'cluster':
        path_noisy_dataset = Path.joinpath(
            consts.PATH_PROJECT_REPORTS_NOISY_DATASETS, 'noisy_dataset_cluster_{}.csv'.format(dataset_id))
        return path_noisy_dataset
    else:
        dict_datasets_map = consts.DICT_DATASETS_MAP_HETE

    if dataset_id in dict_datasets_map:
        return dict_datasets_map.get(dataset_id)
    else:
        return None


def load_preprocessed_dataset(dataset_name, categorical_encoding):

    path_dataset = Path.joinpath(consts.PATH_PROJECT_DIR, 'data', 'processed',
                                 '{}_{}_preprocessed.csv'.format(dataset_name, categorical_encoding))

    if path_dataset is not None:
        df = pd.read_csv(path_dataset)
        logger.info('Loaded dataset: {}'.format(path_dataset))
        if 'label' not in df.columns.to_list():
            y_label = df.iloc[:, -1]
            df_features = df.iloc[:, :-1]
        else:
            y_label = df['label']
            df_features = df.drop('label', axis=1)
        return path_dataset, df_features, y_label
    else:
        raise ValueError("Dataset {} identifier is not recognized!".format(dataset_name))


def normalize_dataframe(x_train, x_test, type_scaler='standard'):

    if type_scaler == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test


def load_dataset_with_noise(dataset_name, noise_type, other_path=None):
    # path_dataset = get_dataset_path_from_id(dataset_name, noise_type)
    # if other_path is not None:
    #     path_dataset = other_path
    file_name=f'noisy_dataset_{noise_type}_{dataset_name}.csv'
    path_dataset=os.path.join(consts.PATH_PROJECT_DIR, 'reports', 'noisy_dataset',file_name)

    if path_dataset is not None:
        df = pd.read_csv(path_dataset)
        logger.info('Loaded dataset: {}'.format(path_dataset))
        y_label = df['label']
        features = df.drop('label', axis=1)
        return path_dataset, features, y_label
    else:
        raise ValueError("Dataset {} identifier is not recognized!".format(dataset_name))


def load_dataset_augmented(with_noise, idx, dataset_name, type_sampling, oversampler, noise_type=None):

    if with_noise:
        generic_name = '{}_{}_{}_{}'.format(dataset_name, noise_type, type_sampling, oversampler)
        file_name = '{}_seed_{}'.format(generic_name, idx)
        train = pd.read_csv(os.path.join(consts.PATH_PROJECT_CTGAN_NOISE,
                                         'train_{}.csv'.format(file_name)))
        test = pd.read_csv(os.path.join(consts.PATH_PROJECT_CTGAN_NOISE,
                                        'test_{}.csv'.format(file_name)))
    else:
        generic_name = '{}_{}_{}'.format(dataset_name, type_sampling, oversampler)
        file_name = '{}_seed_{}'.format(generic_name, idx)
        train = pd.read_csv(os.path.join(consts.PATH_PROJECT_CTGAN_NO_NOISE,
                                         'train_{}.csv'.format(file_name)))
        test = pd.read_csv(os.path.join(consts.PATH_PROJECT_CTGAN_NO_NOISE,
                                        'test_{}.csv'.format(file_name)))

    x_train_scaled, y_train = train.drop(['label'], axis=1), train['label']
    x_test_scaled, y_test = test.drop(['label'], axis=1), test['label']

    return file_name, x_train_scaled, x_test_scaled, y_train, y_test


def load_images(dataset_name, noise_type, other_path=None):
    def extract_number(filename):
        try:
            # Extract the number of name of file
            return int(''.join(filter(str.isdigit, filename)))
        except ValueError:
            return 0

    path_dataset = get_image_path_from_id(dataset_name, noise_type)
    if other_path is not None:
        path_dataset = other_path

    if path_dataset is not None:
        file_list = list(glob.iglob(str(path_dataset) + '/**/*.png', recursive=True))
        # Sort list of files using extract_number function as key
        images = sorted(file_list, key=extract_number)
        # We convert it to an array with that shape so that we can use it with RandomUnderSampler to balance the data
        images = np.array(images).reshape(-1, 1)

        return images
    else:
        raise ValueError("Dataset {} identifier is not recognized!".format(dataset_name))


def load_images_txt(dataset_name, noise_type, other_path=None):
    def extract_number(filename):
        try:
            # Extract the number of name of file
            return int(''.join(filter(str.isdigit, filename)))
        except ValueError:
            return 0

    path_dataset = get_image_path_from_id(dataset_name, noise_type)
    if other_path is not None:
        path_dataset = other_path

    if path_dataset is not None:
        file_list = list(glob.iglob(str(path_dataset) + '/**/*.txt', recursive=True))
        # Sort list of files using extract_number function as key
        images = sorted(file_list, key=extract_number)
        # We convert it to an array with that shape so that we can use it with RandomUnderSampler to balance the data
        images = np.array(images).reshape(-1, 1)

        return images
    else:
        raise ValueError("Dataset {} identifier is not recognized!".format(dataset_name))


def load_images_interpretability(dataset_name, noise_type):

    def extract_number(filename):
        try:
            # Extract the number from the file name
            return int(''.join(filter(str.isdigit, filename)))
        except ValueError:
            # In case of error, return 0 (you can adjust this if necessary)
            return 0
        
    path_dataset = get_image_path_from_id_interpretability(dataset_name, noise_type)
    if path_dataset is not None:
        file_list = list(glob.iglob(str(path_dataset) + '/**/*.png', recursive=True))
        # Sort list of files using extract_number function as key
        images = sorted(file_list, key=extract_number)
        print(images)
        # We convert it to an array with that shape so that we can use it with RandomUnderSampler to balance the data
        images = np.array(images).reshape(-1, 1)

        return images
    else:
        raise ValueError("Dataset {} identifier is not recognized!".format(dataset_name))
