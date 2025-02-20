import gc
from keras.wrappers.scikit_learn import KerasClassifier
# from keras.wrappers.sklearn import KerasClassifier
# from scikeras.wrappers import KerasClassifier
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout,BatchNormalization
from utils.loader import load_images
from pathlib import Path
import logging
import argparse
from utils.img_pre import Interpolation
from utils.loader import load_dataset_with_noise
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.under_sampling import RandomUnderSampler
import cv2
import numpy as np
from multiprocessing import cpu_count
from sklearn import model_selection
import coloredlogs
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.applications import ResNet50
import utils.consts as consts
from keras import layers
from tensorflow.keras import Model, layers
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score, recall_score, multilabel_confusion_matrix
from keras.optimizers import Adam, RMSprop
import os
import tensorflow as tf
from sklearn.metrics import make_scorer
import time
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint  
from tensorflow.keras.models import load_model

# from skopt import BayesSearchCV
# from skopt.space import Real, Categorical, Integer
# from ray import tune
# from ray.tune.sklearn import TuneGridSearchCV
import pandas as pd
import math

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def compute_classification_prestations(y_true: np.array, y_pred: np.array) -> (float, float, float, float):
    # print(len(np.unique(y_true)))
    if len(np.unique(np.argmax(y_true, axis=1))) == 2:
        tn, fp, fn, tp = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1)).ravel()

        acc_val = accuracy_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
        specificity_val = tn / (tn + fp)
        recall_val = recall_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
        # fpr, tpr, threshold = metrics.roc_curve(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
        roc_val = roc_auc_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))

    else:
        cm = multilabel_confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))[0]
        print(cm)
        num_classes = cm.shape[0]
        specificity_val = 0

        for i in range(num_classes):
            true_negatives = sum(cm[j, j] for j in range(num_classes)) - cm[i, i]
            false_positives = sum(cm[:, i]) - cm[i, i]
            specificity_val += true_negatives / (true_negatives + false_positives)

        specificity_val /= num_classes

        acc_val = accuracy_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
        recall_val = recall_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), average='macro')
        roc_val = roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro')

    return acc_val, specificity_val, recall_val, roc_val


def roc_auc(y_true, y_pred):
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.float32)


def create_optimizer(optimizer_name, learning_rate):
    if optimizer_name == 'adam':
        return Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        return RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported.")


def return_cnn_param_grid():
    param_grid = {
        'filters': [8, 16, 32, 64],
        'kernel_size': [(3, 3), (4, 4), (5, 5)],
        'pool_size': [(2, 2), (3, 3)],
        'optimizer': ['adam', 'rmsprop'],
        'dense_units': [64, 128, 256],  # Prueba diferentes tamaños de capas densas
        'learning_rate': [0.001, 0.01, 0.1],  # Prueba diferentes tasas de aprendizaje iniciales
        'dropout_rate': [0.2, 0.4, 0.5]}

    param_rand = {
        'filters': [8, 16, 32, 64],
        'kernel_size': [(3, 3), (4, 4), (5, 5)],
        'pool_size': [(2, 2), (3, 3)],
        'optimizer': ['adam', 'rmsprop'],
        'dense_units': [32, 64, 128],  # Prueba diferentes tamaños de capas densas
        'learning_rate': uniform(0.01, 0.1),  # Prueba diferentes tasas de aprendizaje iniciales
        'dropout_rate': [0.1, 0.2, 0.4]}

    # param_dist = {
    #     'optimizer': Categorical(['adam', 'sgd', 'rmsprop']),
    #     'filters': Integer(16, 64),
    #     'kernel_size': Categorical([3, 4, 5]),  # Valores enteros en lugar de un rango
    #     'pool_size': Categorical([2, 3]),  # Valores enteros 2 y 3
    #     'learning_rate': Real(1e-6, 1e-2, prior='log-uniform'),
    #     'dropout_rate': Real(0.2, 0.5),
    #     'dense_units': Integer(64, 256),
    # }

    # param_tune = {
    #     'optimizer': ['adam', 'rmsprop'],
    #     'filters': [8, 16, 32, 64],
    #     'kernel_size': [3, 4, 5],
    #     'pool_size': [(2, 2), (3, 3)],
    #     'learning_rate': [0.001, 0.01, 0.1],  # Define learning_rate as a list
    #     'dropout_rate': [0.2, 0.5],
    #     'dense_units': [64, 128, 256],
    # }

    return param_rand


def create_cnn_model(optimizer='adam',
                     filters=8,
                     kernel_size=(3, 3),
                     pool_size=(2, 2),
                     learning_rate=0.01,
                     dropout_rate=0.5,
                     dense_units=64
                     ):

    model = Sequential()

    # Capa de convolución con 'filters' filtros, cada uno de tamaño 'kernel_size', activación ReLU
    # model.add(Conv2D(filters, kernel_size, input_shape=(32, 32, 1), activation='relu')) # grayscale
    model.add(Conv2D(filters, kernel_size, input_shape=(32, 32, 3), activation='relu')) # color
    model.add(BatchNormalization())  # Capa de Batch Normalization

    # Capa de pooling para reducir la dimensionalidad
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(BatchNormalization())  # Capa de Batch Normalization

    # # # Capa de convolución con 'filters' filtros, cada uno de tamaño 'kernel_size', activación ReLU
    model.add(Conv2D(filters, kernel_size, activation='relu'))
    model.add(BatchNormalization())  # Capa de Batch Normalization

    # # Capa de pooling para reducir la dimensionalidad
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(BatchNormalization())  # Capa de Batch Normalization

    # Aplanar la salida para conectarla a capas densas
    model.add(Flatten())

    # Capa densa con 128 neuronas y activación ReLU
    model.add(Dense(dense_units, activation='relu'))
    model.add(BatchNormalization())  # Capa de Batch Normalization

    # Capa de dropout para regularización
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())  # Capa de Batch Normalization

    if len(np.unique(y_label)) == 2:
        loss_function = 'binary_crossentropy'
        output_neurons = len(np.unique(y_label))
    else:
        loss_function = 'categorical_crossentropy'
        output_neurons = len(np.unique(y_label))

    model.add(Dense(output_neurons, activation='softmax'))
    optimizer_obj = create_optimizer(optimizer, learning_rate)
    model.compile(optimizer=optimizer_obj, loss=loss_function, metrics=['accuracy'])

    return model


def BinaryCrossEntropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
    term_1 = y_true * np.log(y_pred + 1e-7)
    return -np.mean(term_0+term_1, axis=0)


def parse_arguments(parser):
    parser.add_argument('--dataset', default='crx', type=str)
    parser.add_argument('--noise_type', default='heterogeneous', type=str)
    parser.add_argument('--type_padding', default='none', type=str)
    parser.add_argument('--padding', default=50, type=int)
    parser.add_argument('--type_interpolation', default='none', type=str)
    parser.add_argument('--interpolation', default=5, type=int)
    parser.add_argument('--n_seeds', default=5, type=int)
    parser.add_argument('--n_jobs', default=4, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--cuda', default=False, type=bool)
    return parser.parse_args()


def add_padding(image, mode, pad=50):
    """
    modes = ["wrap", "mean", "reflect", "symmetric", "constant", "linear_ramp"]
    """
    print(image.shape)
    modes = ["wrap", "mean", "reflect", "symmetric", "constant", "linear_ramp"]

    # Loop through the padding methods and apply them to the image
    # pad_width=((top, bottom), (left, right))
    if mode == "constant":
        img_padded = np.pad(
            image,
            pad_width=((pad, pad), (pad, pad), (0, 0)),
            mode=mode,
            constant_values=0,
        )
    else:
        img_padded = np.pad(image, pad_width=((pad, pad), (pad, pad), (0, 0)), mode=mode)

    return img_padded


if __name__ == "__main__":

    start_time = time.time()

    parser = argparse.ArgumentParser(description='train cnn')
    args = parse_arguments(parser)

    n_procs = cpu_count()
    n_jobs = n_procs if args.n_jobs == -1 or args.n_jobs > n_procs else args.n_jobs
    logger.info('n_jobs for train_cnn: {}'.format(n_jobs))

    images = load_images(args.dataset, args.noise_type)
    path_dataset, features, y_label = load_dataset_with_noise(args.dataset, args.noise_type)

    list_acc_values = []
    list_specificity_values = []
    list_recall_values = []
    list_auc_values = []
    print(y_label)
    rus = RandomUnderSampler(sampling_strategy='all', random_state=42)
    X_res, y_res = rus.fit_resample(images, y_label)
    print('Resampled dataset shape %s' % y_res.value_counts())

    # imgs=[]
    # for i in range(len(X_res)):
    #     imgs.append(cv2.cvtColor(cv2.imread(X_res[i][0]), cv2.COLOR_BGR2GRAY))
    #     print(X_res[i][0])

    # print(imgs[0].shape)

    imgs = []

    for i in range(len(X_res)):
        # imgs.append(cv2.cvtColor(cv2.imread(X_res[i][0]), cv2.COLOR_BGR2GRAY)) # grayscale
        imgs.append((cv2.imread(X_res[i][0]))/255)  # color
        # print(X_res[i][0])

    # print(imgs[0].shape)

    imagenes_resize = []
    for imagen in range(len(imgs)):
        img_i = imgs[imagen]

        if args.type_padding != 'none':
            img_i = add_padding(img_i, mode=args.type_padding, pad=args.padding)

        if args.type_interpolation != 'none':
            interp = Interpolation(img_i, scale=args.interpolation, type_interpolation=args.type_interpolation)
            img_i = interp.compute_interpolation()

        imagenes_resize.append(cv2.resize(img_i, (32, 32)))

        # imagenes_resize.append(cv2.resize(imgs[imagen], (32, 32)))

    loss_scorer = make_scorer(BinaryCrossEntropy, greater_is_better=False)
    model = KerasClassifier(build_fn=create_cnn_model, verbose=1)
    print(model.get_params().keys())

    learn_control = ReduceLROnPlateau(monitor='val_loss', patience=20, verbose=1, factor=0.1, min_lr=1e-20)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=200)

    dict_cnn_param_grid = return_cnn_param_grid()
    print(len(np.unique(y_label)))

    for idx in range(args.n_seeds):
        model = KerasClassifier(build_fn=create_cnn_model, verbose=args.verbose)
        X_train, X_test, Y_train, Y_test = train_test_split(imagenes_resize, y_res, stratify=y_res, test_size=0.2, random_state=idx)
        X_train, X_val, y_train, y_val = train_test_split(X_train, Y_train, stratify=Y_train, test_size=0.15, random_state=idx)

        X_train = np.array(X_train)
        X_test = np.array(X_test)
        X_val = np.array(X_val)

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

        # train_generator = ImageDataGenerator(
        #     zoom_range=2,
        #     rotation_range=90,
        #     horizontal_flip=True,
        #     vertical_flip=True,
        # )

        batch_size = args.batch_size
        csv_file_path = str(Path.joinpath(consts.PATH_PROJECT_SAVE_MODEL,
                                          f'weights_best_cnn_{args.noise_type}_{args.dataset}_batch_{batch_size}_idx{idx}_tune.hdf5')
                            )

        # # #filepath="weights"+str(semilla)+ ".best.hdf5"
        checkpoint = ModelCheckpoint(csv_file_path, monitor='val_loss', verbose=args.verbose, save_best_only=False, mode='min')


    # Configura BayesSearchCV
    #     grid = BayesSearchCV(
    #         estimator=model,  # Tu modelo Keras
    #         search_spaces=dict_cnn_param_grid,
    #         n_jobs=n_jobs,
    #         cv=3,
    #         n_iter=10,  # Número de iteraciones de búsqueda
    #         random_state=42  # Semilla para reproducibilidad
    #     )
        grid = RandomizedSearchCV(estimator=model,
                                  param_distributions=dict_cnn_param_grid,
                                  n_iter=80,
                                  cv=5,
                                  n_jobs=n_jobs,
                                  verbose=args.verbose)

        # grid = TuneGridSearchCV(
        #     estimator=model,
        #     param_grid=dict_cnn_param_grid,
        #     n_jobs=n_jobs,
        #     cv=5,
        #     verbose=args.verbose  # Nivel de verbosidad
        # )
        # grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=1,n_jobs=-1)

        grid_result = grid.fit(X_train, Y_train, steps_per_epoch=math.ceil(X_train.shape[0] / batch_size))



    # # Muestra los resultados
    # print("Mejor: %f usando %s" % (grid_result.best_score_, grid_result.best_params_))

    # # Evalúa el modelo con los mejores hiperparámetros en un conjunto de prueba
    # best_model = grid_result.best_estimator_
    # accuracy = best_model.score(X_test, Y_test)
    # print("Exactitud en conjunto de prueba: %f" % accuracy)
        best_params = grid_result.best_params_
        print(best_params)
        model = create_cnn_model(
            optimizer=best_params['optimizer'],
            filters=best_params['filters'],
            kernel_size=best_params['kernel_size'],
            pool_size=best_params['pool_size'],
            learning_rate=best_params['learning_rate'],
            dropout_rate=best_params['dropout_rate'],
            dense_units=best_params['dense_units']
        )

        model.fit(X_train, Y_train, batch_size=batch_size,

        steps_per_epoch = math.ceil(
            X_train.shape[0] / batch_size),
            epochs=1000,
            validation_data=(X_val, Y_val),
            callbacks=[learn_control, early, checkpoint]
        )

        # model=create_cnn_model()
        model.load_weights(csv_file_path)
        model.save(str(Path.joinpath(consts.PATH_PROJECT_SAVE_MODEL, f"modelo_best_cnn_{args.noise_type}_{args.dataset}_batch_{batch_size}_idx{idx}_tune.h5")))

        Y_pred = model.predict(X_test)

        acc_val, specificity_val, recall_val, roc_auc_val = compute_classification_prestations(Y_test, Y_pred)


        # Accuracy=accuracy_score(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1))
        # Recall=recall_score(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1))
        # tn, fp, fn, tp = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1)).ravel()
        # Especificidad = tn / (tn + fp)
        # fpr, tpr, threshold = metrics.roc_curve(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1))
        # roc_auc = metrics.auc(fpr, tpr)


        # print(Accuracy, Recall, Especificidad, roc_auc)


        # list_acc_values.append(Accuracy)
        # list_specificity_values.append(Especificidad)
        # list_recall_values.append(Recall)
        # list_auc_values.append(roc_auc)

        print(acc_val, recall_val, specificity_val, roc_auc_val)

        list_acc_values.append(acc_val)
        list_specificity_values.append(specificity_val)
        list_recall_values.append(recall_val)
        list_auc_values.append(roc_auc_val)


        #pickle.dump(clf_model, open(str(Path.joinpath(consts.PATH_PROJECT_MODELS,
                                                    #'model_clf_{}.sav'.format(generic_name_partition))), 'wb'))

    end_time = time.time()  # Registra el tiempo de finalización
    elapsed_time = end_time - start_time  # Calcula el tiempo transcurrido
    print(f"Tiempo de ejecución: {elapsed_time} segundos")

    mean_std_specificity = np.mean(list_specificity_values), np.std(list_specificity_values)
    mean_std_accuracy = np.mean(list_acc_values), np.std(list_acc_values)
    mean_std_recall = np.mean(list_recall_values), np.std(list_recall_values)
    mean_std_auc = np.mean(list_auc_values), np.std(list_auc_values)

    print('accuracy:', mean_std_accuracy)
    print('specificity:', mean_std_specificity)
    print('recall:', mean_std_recall)
    print('AUC:', mean_std_auc)

    exp_name = '{}+{}+{}'.format(args.dataset, 'cnn', args.n_seeds)
    #exp_name = '{}+{}'.format(exp_name, 'fs') if args.flag_fs else exp_name

    new_row_auc = {'model': exp_name,
                   'eval_metric': 'auc',
                   'mean': mean_std_auc[0],
                   'std': mean_std_auc[1]}

    new_row_sensitivity = {'model': exp_name,
                           'eval_metric': 'sensitivity',
                           'mean': mean_std_recall[0],
                           'std': mean_std_recall[1]}

    new_row_specificity = {'model': exp_name,
                           'eval_metric': 'specificity',
                           'mean': mean_std_specificity[0],
                           'std': mean_std_specificity[1]}

    new_row_accuracy = {
        'model': exp_name,
        'eval_metric': 'accuracy',
        'mean': mean_std_accuracy[0],
        'std': mean_std_accuracy[1]
    }

    csv_file_path = str(Path.joinpath(consts.PATH_PROJECT_METRICS, 'metrics_classification_cnn_{}_{}.csv'.format(args.noise_type, args.dataset)))

    if os.path.exists(csv_file_path):
        try:
            df_metrics_classification = pd.read_csv(csv_file_path)
        except pd.errors.EmptyDataError:
            df_metrics_classification = pd.DataFrame()
    else:
        df_metrics_classification = pd.DataFrame()

    df_metrics_classification = pd.concat([df_metrics_classification, pd.DataFrame([new_row_auc])], ignore_index=True)
    df_metrics_classification = pd.concat([df_metrics_classification, pd.DataFrame([new_row_sensitivity])], ignore_index=True)
    df_metrics_classification = pd.concat([df_metrics_classification, pd.DataFrame([new_row_specificity])], ignore_index=True)
    df_metrics_classification = pd.concat([df_metrics_classification, pd.DataFrame([new_row_accuracy])], ignore_index=True)

    # df_metrics_classification = df_metrics_classification.append(new_row_auc, ignore_index=True)
    # df_metrics_classification = df_metrics_classification.append(new_row_sensitivity, ignore_index=True)
    df_metrics_classification.to_csv(csv_file_path, index=False)

