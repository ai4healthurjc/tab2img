import argparse
from pathlib import Path
from utils.dissimilarity import compute_corr_mixed_dataset_2
from utils.preprocessing import identify_feature_type
from utils.loader import load_dataset_with_noise
from utils.plotter import plot_corr_mixed_matrix
import utils.consts as consts
import matplotlib.pyplot as plt
import seaborn as sns


def parse_arguments(parser):
    parser.add_argument('--dataset', default='crx', type=str)
    parser.add_argument('--noise_type', default='homogeneous', type=str)

    return parser.parse_args()


parser = argparse.ArgumentParser(description='noise generator experiments')
args = parse_arguments(parser)
path_dataset, features, y_label = load_dataset_with_noise(args.dataset, args.noise_type)
dataset, categorical, numeric = identify_feature_type(features, categorical_threshold=0.05, impute_missing=False)
# scaler = StandardScaler()
# features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

# binarias, numericas = identify_binary_and_numeric_features(features)

df_corr = compute_corr_mixed_dataset_2(features, categorical, numeric)
csv_file_path = str(Path.joinpath(consts.PATH_PROJECT_SAVE_PLOTS, 'Correlation_{}_{}.pdf'.format(args.noise_type, args.dataset)))
plot_corr_mixed_matrix(df_corr, csv_filepath=csv_file_path, figsize=(19, 19), flag_save_figure=True)



# Gráfica de distribución para todas las variables numéricas
#fig, axes = plt.subplots(nrows=1, ncols=len(numeric), figsize=(15, 4))

# for i, var in enumerate(numeric):
#     fig, ax = plt.subplots(figsize=(8, 6))
#     sns.kdeplot(data=dataset[var], ax=ax)
#     ax.set_title(f'Distribución de {var}')
#     fig.savefig(f'grafica_numerica_{var}.png')  # Cambia el nombre del archivo según tus necesidades
#     plt.close()  # Cerrar la figura actual para liberar memoria


# # fig.savefig('numeric_combinadas.png')  # Cambia el nombre del archivo según tus necesidades
# # plt.close()

# # Gráfica de barras para todas las variables categóricas
# # fig, axes = plt.subplots(nrows=1, ncols=len(categorical), figsize=(15, 4))

# for i, var in enumerate(categorical):
#     fig, ax = plt.subplots(figsize=(8, 6))
#     sns.countplot(x=var, data=dataset, ax=ax, color='orange')
#     for p in ax.patches:
#         ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=8)

#     ax.set_title(f'Distribución de {var}')
#     fig.savefig(f'grafica_categorica_{var}.png')  # Cambia el nombre del archivo según tus necesidades
#     plt.close()  # Cerra

# # fig.savefig('categorical_combinadas.png')  # Cambia el nombre del archivo según tus necesidades
# # plt.close()
