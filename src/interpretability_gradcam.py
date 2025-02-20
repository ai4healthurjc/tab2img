import numpy as np
import pandas as pd
import argparse
import math
from pathlib import Path
from multiprocessing import cpu_count
import coloredlogs
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw, ImageFont
import shutil

from utils.loader import load_dataset_with_noise
import tensorflow as tf

from utils.IGTDTransformer_Interpretability import IGTD_Transformer
from utils.preprocessing import identify_feature_type
import utils.feature_selection_bootstrap as fs
import utils.consts as consts
from utils.loader import load_images, load_images_interpretability
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

from tensorflow.keras import Model, layers

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


def parse_arguments(parser):
    parser.add_argument('--dataset', default='crx', type=str)
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--n_jobs', default=4, type=int)
    parser.add_argument('--fs', default='relief', type=str)
    parser.add_argument('--agg_func', default='mean', type=str)
    parser.add_argument('--noise_type', default='homogeneous', type=str)

    return parser.parse_args()

parser = argparse.ArgumentParser(description='noise generator experiments')
args = parse_arguments(parser)

n_procs = cpu_count()
n_jobs = n_procs if args.n_jobs == -1 or args.n_jobs > n_procs else args.n_jobs
logger.info('n_jobs for tab_to_image_fs: {}'.format(n_jobs))

images = load_images(args.dataset, args.noise_type)
images_interpretability = load_images_interpretability(args.dataset, args.noise_type)
path_dataset, features, y_label = load_dataset_with_noise(args.dataset, args.noise_type)

print(y_label)

images_interpretability = np.delete(images_interpretability, 0, axis=0)

rus = RandomUnderSampler(sampling_strategy='all', random_state=42)
X_res, y_res = rus.fit_resample(images, y_label)
X_res_interpretability, y_res_interpretability = rus.fit_resample(images_interpretability, y_label)
print('Resampled dataset shape %s' % y_res.value_counts())
print('Resampled dataset shape %s' % y_res_interpretability.value_counts())

# imgs=[]
# for i in range(len(X_res)):
#     imgs.append(cv2.cvtColor(cv2.imread(X_res[i][0]), cv2.COLOR_BGR2GRAY))
#     print(X_res[i][0])

# print(imgs[0].shape)

imgs=[]
imgs_interpretability=[]

for i in range(len(X_res)):
    imgs.append((cv2.imread(X_res[i][0]))/255)
    imgs_interpretability.append((cv2.imread(X_res_interpretability[i][0]))/255)
    # print(X_res[i][0])

# print(imgs[0].shape)

imagenes_resize = []
imagenes_resize_interpretability = []

for imagen in range(len(imgs)):
    imagenes_resize.append(cv2.resize(imgs[imagen], (32, 32)))
    #imagenes_resize_interpretability.append(cv2.resize(imgs_interpretability[imagen], (32, 32)))


csv_file_path = str(Path.joinpath(consts.PATH_PROJECT_SAVE_MODEL, f'modelo_best_cnn_{args.noise_type}_{args.dataset}_batch_32_idx{args.seed}_tune.h5'))

X_train, X_test, Y_train, Y_test = train_test_split(imagenes_resize, y_res, stratify=y_res, test_size=0.2, random_state=args.seed)
X_train, X_val, y_train, y_val = train_test_split(X_train, Y_train, stratify=Y_train, test_size=0.15, random_state=args.seed)

X_train_interpretability, X_test_interpretability, Y_train_interpretability, Y_test_interpretability = train_test_split(imgs_interpretability, y_res_interpretability, stratify=y_res_interpretability, test_size=0.2, random_state=args.seed)
X_train_interpretability, X_val_interpretability, y_train_interpretability, y_val_interpretability = train_test_split(X_train_interpretability, Y_train_interpretability, stratify=Y_train_interpretability, test_size=0.15, random_state=args.seed)

X_train = np.array(X_train)
X_test = np.array(X_test)
X_val = np.array(X_val)

X_train_interpretability = np.array(X_train_interpretability)
X_test_interpretability = np.array(X_test_interpretability)
X_val_interpretability = np.array(X_val_interpretability)

# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
# X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)

# # # X_test = X_test.reshape(X_test.shape[0], 7, 9, 1)
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
# Y_train=np.array(y_train)
# Y_test=np.array(Y_test)
# Y_val=np.array(y_val)

Y_train = to_categorical(y_train, num_classes=len(np.unique(y_label)))
Y_test = to_categorical(Y_test, num_classes=len(np.unique(y_label)))
Y_val = to_categorical(y_val, num_classes=len(np.unique(y_label)))

Y_train_interpretability = to_categorical(y_train, num_classes=len(np.unique(y_label)))
Y_test_interpretability= to_categorical(Y_test, num_classes=len(np.unique(y_label)))
y_val_interpretability = to_categorical(y_val, num_classes=len(np.unique(y_label)))

model = load_model(csv_file_path)
Y_pred = model.predict(X_test)

model.summary()

# Carga tu conjunto de prueba (imágenes de entrada)
conjunto_prueba = X_test  # Carga tu conjunto de prueba

# Define la capa de interés en tu modelo
capa_interes = model.get_layer('conv2d_1').output  # Reemplaza "nombre_de_la_capa" por el nombre real de la capa de interés

# Obtén el índice de la capa convolucional de interés
#indice_capa = 169  # Índice de la capa convolucional en ResNet50 (conv5_block3_out)

# Crea un nuevo modelo que tenga la capa de interés como salida
grad_model = tf.keras.models.Model(inputs=model.input, outputs=[capa_interes, model.output])

# Crea un nuevo modelo que tenga la capa de interés como salida
#grad_model = tf.keras.models.Model(inputs=model.input, outputs=[capa_interes, model.output])

# Lista para almacenar los mapas de calor generados
mapas_calor = []
pred_index=None
# Itera sobre todas las imágenes en el conjunto de prueba
for imagen_input in conjunto_prueba:
    # Preprocesa la imagen de entrada según las necesidades de tu modelo
    #imagen_input = [...]  # Preprocesa tu imagen según las necesidades de tu modelo

    # Expande las dimensiones de la imagen para que coincida con la forma de entrada del modelo
    img_array = np.expand_dims(imagen_input, axis=0)

    # Realiza la predicción en el modelo de interés
    #prediccion = modelo_interes.predict(imagen_input_expandida)
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    #print(grads)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    #print(heatmap)
    #heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    mapas_calor.append(heatmap)


gradcam_filepath = str(Path.joinpath(consts.PATH_PROJECT_SAVE_HEATMAPS, f'heatmaps_{args.noise_type}_{args.dataset}'))

if os.path.exists(gradcam_filepath):
    shutil.rmtree(gradcam_filepath)
os.mkdir(gradcam_filepath)

def plot_attention_map(att_maps, images):
   
    for att_map_idx in range(len(att_maps)):
        filename = f"_{att_map_idx}_image.png"
        filepath = os.path.join(gradcam_filepath, filename)
        att_map = np.array(att_maps[att_map_idx])

        # Asegurarse de que att_map tenga un rango válido
        att_map_range = att_map.max() - att_map.min()
        att_map_normalize = (att_map - att_map.min()) / (att_map_range + 1e-10)  # Añadir pequeño epsilon para evitar división por cero

        # Redimensionar el mapa de atención
        att_map_resize = cv2.resize(att_map_normalize, (images[att_map_idx].shape[1], images[att_map_idx].shape[0]))

        # Crear una versión en color del mapa de atención con un esquema de colores 'COLORMAP_JET'
        att_map_rgb = cv2.applyColorMap(np.uint8(att_map_resize * 255), cv2.COLORMAP_JET)

        # Asegurarse de que los valores estén en el rango correcto (0-255)
        # att_map_rgb = np.clip(att_map_rgb, 0, 255).astype(np.uint8)

        # Convertir la matriz de la imagen al mismo tipo de datos que att_map_rgb
        image_for_combination = (images[att_map_idx] * 255).astype(att_map_rgb.dtype)

        print(att_map_rgb.shape)
        print(image_for_combination.shape)

        # Combinar la imagen original con el mapa de atención
        output_image = cv2.addWeighted(image_for_combination, 0.7, att_map_rgb, 0.3, 0)

        print(att_map_idx)
        print((np.argmax(Y_pred[att_map_idx]), np.argmax(Y_test[att_map_idx])))
        #text = f'Label Prediction: {np.argmax(Y_pred[att_map_idx])}, Real label: {np.argmax(Y_test[att_map_idx])}'
        #cv2.putText(output_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Guardar la imagen combinada utilizando cv2.imwrite
        cv2.imwrite(filepath, output_image)

        # # Convertir el array de NumPy a un objeto Image
        # output_image_pil = Image.fromarray(output_image)

        # # Crear el objeto ImageDraw
        # draw = ImageDraw.Draw(output_image_pil)
        # # Crear un objeto ImageDraw para agregar texto

        # # Configurar la fuente y el tamaño del texto
        # font = ImageFont.load_default()

        # # Configurar la posición del texto en la imagen
        # text_position = (10, 10)

        # # Crear el texto que refleje la salida de print(np.argmax(Y_pred[att_map]), np.argmax(Y_test[att_map]))
        # text_to_display = f'Label Prediction: {np.argmax(Y_pred[att_map_idx])}, Real label: {np.argmax(Y_test[att_map_idx])}'
        # # Agregar el texto a la imagen
        # draw.text(text_position, text_to_display, fill=(255, 255, 255), font=font)

        # Guardar la imagen combinada utilizando cv2.imwrite
        cv2.imwrite(filepath, output_image)





        # # Mostrar la imagen original y la imagen combinada con el mapa de atención
        # plt.imshow(images[att_map_idx])
        # plt.title('Imagen Original')
        # plt.axis('off')
        # plt.show()

        # plt.imshow(output_image)
        # plt.title('Imagen con Mapa de Atención')
        # plt.axis('off')
        # plt.show()



# attn_model = Model(inputs=model.input, outputs=model.get_layer('locally_connected2d').output)
plot_attention_map(mapas_calor, X_test_interpretability)


# # Utilizamos el nuevo modelo para hacer predicciones en las imágenes de test cuya salida será el mapa de atención
# attn_preds = attn_model.predict(X_test)