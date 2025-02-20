import math
import argparse
from pathlib import Path
import coloredlogs
import logging

from utils.loader import load_dataset_with_noise
from utils.preprocessing import identify_feature_type
from utils.IGTDTransformer import IGTDTransformer
import utils.consts as consts

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def parse_arguments(parser):
    parser.add_argument('--dataset', default='crx', type=str)
    return parser.parse_args()


parser = argparse.ArgumentParser(description='noise generator experiments')
args = parse_arguments(parser)

path_dataset, features, y_label = load_dataset_with_noise(args.dataset)
dataset, categorical, numeric = identify_feature_type(features, categorical_threshold=0.1, impute_missing=False)

num_features = len(features.columns)  # Número total de características en tu conjunto de datos

# Calcula el número de filas y columnas de la imagen basándote en el número de características
# Puedes elegir una estrategia para determinar las dimensiones, por ejemplo, tomar la raíz cuadrada o cualquier otra.
print(num_features)
num_row = int(math.sqrt(num_features))
#num_row=38
num_col = int(math.ceil(num_features / num_row))
#num_col=num_row
print(num_row, num_col)

logger.info(f"Tu dataset {args.dataset} tiene un total de {num_features} features: Se generarán imágenes de tamaño - ({num_col},{num_row})")

#for i in range(features.shape[0]):
csv_file_path = str(Path.joinpath(consts.PATH_PROJECT_TAB_TO_IMAGE, f'image_{args.dataset}'))


best_noisy_data_generator = IGTDTransformer(num_row=num_row, num_col=num_col, save_image_size=1, max_step=1000, val_step=100, result_dir=csv_file_path, numericas=numeric, categoricas=categorical, error='abs')
noisy_data = best_noisy_data_generator.transform(features)
    #noisy_data_with_labels = pd.concat([noisy_data, pd.Series(y_label, name='y_label')], axis=1)

    #noisy_data_with_labels.to_csv(csv_file_path, index=False)