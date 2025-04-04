import os
import numpy as np
import shutil
import cv2
import utils.consts as cons
import utils.loader as loader


def create_dataset_from_txt( dataset_id, noise_type, path_images, path_dataset, path_noise=None):
    """
    :param dataset_id: name of the dataset to explore
    :param noise_type: type of noise added
    :param path_images: where images are stored
    :param path_dataset: where we want to create our dataset
    :param path_noise: where noisy datasets augmented are stored
    """
    if path_noise is None:
        images_paths = loader.load_images_txt(dataset_id, noise_type)
        _, _, labels = loader.load_dataset_with_noise(dataset_id, noise_type)
    else:
        images_paths = loader.load_images_txt(dataset_id, noise_type, path_images)
        _, _, labels = loader.load_dataset_with_noise(dataset_id, noise_type, path_noise)

    print('Resampled dataset shape %s' % labels.value_counts())

    fullpath_images_resampled = [image[0] for image in images_paths]
    name_images_resampled = [image[0].split('/')[-1].split('_')[1] for image in images_paths]
    name_images_resampled = [f'_{i}_image.png' for i in name_images_resampled]
    y_resampled = list(labels)

    os.makedirs(path_dataset)
    for label in list(set(y_resampled)):
        os.makedirs(os.path.join(path_dataset, str(label)))
        for n, image_label in enumerate(y_resampled):
            if image_label == label:
                with open(fullpath_images_resampled[n], 'r') as file:
                    rows = file.readlines()
                pixel_values = []
                for row in rows:
                    row_value = row.strip().split()
                    pixel_values.append([float(valor) for valor in row_value])
                imagen_array = np.array(pixel_values)
                cv2.imwrite(os.path.join(os.path.join(path_dataset, str(label)), name_images_resampled[n]),
                            imagen_array)


def create_datasets(path, dataset_name, noise_type, file_name=None, interpretability=False):

    if interpretability:
        path_images = os.path.join(path, 'data_interpretability')
        path_dataset_png = os.path.join(path, 'dataset_from_png_interpretability')

        if file_name is None:
            if not os.path.exists(path_dataset_png):
                create_dataset_from_png(dataset_name, noise_type, path_images, path_dataset_png,
                                        interpretability=interpretability)

        else:
            path_noise = os.path.join(cons.PATH_PROJECT_CTGAN_NOISE, f'{file_name}.csv')
            if not os.path.exists(path_dataset_png):
                create_dataset_from_png(dataset_name, noise_type, path_images, path_dataset_png, path_noise)

        return path_dataset_png
    else:

        path_images = os.path.join(path, 'data')
        path_dataset_png = os.path.join(path, 'dataset_from_png')
        path_dataset_txt = os.path.join(path, 'dataset_from_txt')

        if file_name is None:
            if not os.path.exists(path_dataset_png):
                create_dataset_from_png(dataset_name, noise_type, path_images, path_dataset_png)
            if not os.path.exists(path_dataset_txt):
                create_dataset_from_txt(dataset_name, noise_type, path_images, path_dataset_txt)
        else:
            path_noise = os.path.join(cons.PATH_PROJECT_CTGAN_NOISE, f'{file_name}.csv')
            if not os.path.exists(path_dataset_png):
                create_dataset_from_png(dataset_name, noise_type, path_images, path_dataset_png, path_noise)
            if not os.path.exists(path_dataset_txt):
                create_dataset_from_txt(dataset_name, noise_type, path_images, path_dataset_txt, path_noise)

        return path_dataset_txt, path_dataset_png



def create_dataset_from_png(dataset_id, noise_type, path_images, path_dataset, path_noise=None, interpretability=False):
    """
    :param dataset_id: name of the dataset to explore
    :param noise_type: type of noise added
    :param path_images: where images are stored
    :param path_dataset: where we want to create our dataset
    :param path_noise: where noisy datasets augmented are stored
    :param interpretability: if interpretability is applied or not
    """

    if interpretability:
        images_paths = loader.load_images_interpretability(dataset_id, noise_type)
        _, _, labels = loader.load_dataset_with_noise(dataset_id, noise_type)
    else:
        if path_noise is None:
            images_paths = loader.load_images(dataset_id, noise_type)
            _, _, labels = loader.load_dataset_with_noise(dataset_id, noise_type)
        else:
            images_paths = loader.load_images(dataset_id, noise_type, path_images)
            _, _, labels = loader.load_dataset_with_noise(dataset_id, noise_type, path_noise)

    images_resampled_name = [image[0].split('/')[-1] for image in images_paths]
    y_resampled = list(labels)

    os.makedirs(path_dataset)
    for label in list(set(y_resampled)):
        os.makedirs(os.path.join(path_dataset, str(label)))
        for n, image_label in enumerate(y_resampled):
            if image_label == label:
                origin_path = os.path.join(path_images, images_resampled_name[n])
                final_path_aux = os.path.join(path_dataset, str(label))
                final_path = os.path.join(final_path_aux, images_resampled_name[n])
                if origin_path != final_path:
                    shutil.copy(str(origin_path), str(final_path))
                else:
                    print(f"File {images_resampled_name[n]} already in the destination folder.")


