import os
import pandas as pd
from IGTD_Functions import min_max_transform, table_to_image, select_features_by_variation
import coloredlogs
import logging
from loader import load_dataset_with_noise
import argparse
from multiprocessing import cpu_count


logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def parse_arguments(parser):
    parser.add_argument('--dataset', default='ionos', type=str)
    parser.add_argument('--noise_type', default='homogeneous', type=str)
    parser.add_argument('--n_boots', default=100, type=int)
    parser.add_argument('--n_jobs', default=4, type=int)
    parser.add_argument('--max_steps', default=1000, type=int)
    parser.add_argument('--val_step', default=100, type=int)
    parser.add_argument('--fs', default='relief', type=str)
    parser.add_argument('--agg_func', default='mean', type=str)
    parser.add_argument('--igtd_err', default='abs', type=str)
    return parser.parse_args()


parser = argparse.ArgumentParser(description='noise generator experiments')
args = parse_arguments(parser)

n_procs = cpu_count()
n_jobs = n_procs if args.n_jobs == -1 or args.n_jobs > n_procs else args.n_jobs
logger.info('n_jobs for tab_to_image_fs: {}'.format(n_jobs))

num_row = 12    # Number of pixel rows in image representation
num_col = 11    # Number of pixel columns in image representation
num = num_row * num_col # Number of features to be included for analysis, which is also the total number of pixels in image representation
save_image_size = 3 # Size of pictures (in inches) saved during the execution of IGTD algorithm.
max_step = 2000    # The maximum number of iterations to run the IGTD algorithm, if it does not converge.
val_step = 300  # The number of iterations for determining algorithm convergence. If the error reduction rate
                # is smaller than a pre-set threshold for val_step itertions, the algorithm converges.

# Import the example data and linearly scale each feature so that its minimum and maximum values are 0 and 1, respectively.
# data = pd.read_csv('../Data/Example_Gene_Expression_Tabular_Data.txt', low_memory=False, sep='\t', engine='c',
#                    na_values=['na', '-', ''], header=0, index_col=0)

path_dataset, data, y_label = load_dataset_with_noise(args.dataset, args.noise_type)

# Select features with large variations across samples
id = select_features_by_variation(data, variation_measure='var', num=num)
data = data.iloc[:, id]
# Perform min-max transformation so that the maximum and minimum values of every feature become 1 and 0, respectively.
norm_data = min_max_transform(data.values)
norm_data = pd.DataFrame(norm_data, columns=data.columns, index=data.index)

# Run the IGTD algorithm using (1) the Euclidean distance for calculating pairwise feature distances and pariwise pixel
# distances and (2) the absolute function for evaluating the difference between the feature distance ranking matrix and
# the pixel distance ranking matrix. Save the result in Test_1 folder.

# fea_dist_method = 'Euclidean'
# image_dist_method = 'Euclidean'
# error = 'abs'
# result_dir = '../Results/Table_To_Image_Conversion/Test_1'
# os.makedirs(name=result_dir, exist_ok=True)
# table_to_image(norm_data,
#                [num_row, num_col],
#                fea_dist_method,
#                image_dist_method,
#                save_image_size,
#                max_step, val_step, result_dir, error)

# Run the IGTD algorithm using (1) the Pearson correlation coefficient for calculating pairwise feature distances,
# (2) the Manhattan distance for calculating pariwise pixel distances, and (3) the square function for evaluating
# the difference between the feature distance ranking matrix and the pixel distance ranking matrix.
# Save the result in Test_2 folder.
fea_dist_method = 'Pearson'
image_dist_method = 'Manhattan'
error = 'squared'
norm_data = norm_data.iloc[:, :800]
result_dir = '../Results/Table_To_Image_Conversion/Test_2'
os.makedirs(name=result_dir, exist_ok=True)
table_to_image(norm_data, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
               max_step, val_step, result_dir, error)
