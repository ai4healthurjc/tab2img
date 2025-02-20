import argparse
from pathlib import Path
import utils.consts as consts
from utils.loader import load_by_dataset_name
from utils.preprocessing import count_null_values, remove_null_variables, \
    preprocess_categorical_features_different, save_dataset, identify_type_features,remove_unique_variables


def parse_arguments(parser):
    parser.add_argument('--dataset', default='steno', type=str)
    parser.add_argument('--type_dataset', default='binary', type=str)
    parser.add_argument('--threshold_null_values', default=0.6, type=float)
    parser.add_argument('--categorical_encoding', default='count', type=str)
    return parser.parse_args()

cmd_parser = argparse.ArgumentParser(description='tab2img experiments')
args = parse_arguments(cmd_parser)

path_dataset, df_features, y_label = load_by_dataset_name(args.dataset, args.type_dataset)
dataset_without_null, y_label = remove_null_variables(df_features, y_label, args.threshold_null_values)
dataset_preprocess = preprocess_categorical_features_different(dataset_without_null, args.categorical_encoding)
dataset_preprocess = remove_unique_variables(dataset_preprocess)

if args.dataset == 'fram':
    dataset_preprocess['smoker'].replace([2144, 2094], [0, 1], inplace=True)

csv_path_save = str(Path.joinpath(consts.PATH_PROJECT_DATA_PREPROCESSED,
                                  '{}_{}_preprocessed.csv'.format(args.dataset, args.categorical_encoding)))
save_dataset(args.dataset, dataset_preprocess, y_label, csv_path_save)

