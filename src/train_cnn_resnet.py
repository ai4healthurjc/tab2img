from keras.wrappers.scikit_learn import KerasClassifier
#from scikeras.wrappers import KerasClassifier
from keras import backend as K
import gc
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout,BatchNormalization
from utils.loader import load_images
from pathlib import Path
import utils.consts as consts
import coloredlogs
import logging
import argparse
from utils.loader import load_dataset_with_noise
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV, train_test_split
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.applications import ResNet50
import utils.consts as consts
from keras import layers
from tensorflow.keras import Model, layers
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import model_selection   
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from keras.optimizers import Adam, RMSprop  # Importa los optimizadores que necesitas

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

def parse_arguments(parser):
    parser.add_argument('--dataset', default='crx', type=str)
    return parser.parse_args()

#start_time = time.time()  # Registra el tiempo de inicio

parser = argparse.ArgumentParser(description='noise generator experiments')
args = parse_arguments(parser)

images=load_images(args.dataset)
path_dataset, features, y_label = load_dataset_with_noise(args.dataset)

rus = RandomUnderSampler(sampling_strategy='all',random_state=42)
X_res, y_res = rus.fit_resample(images, y_label)
print('Resampled dataset shape %s' % y_res.value_counts())

# imgs=[]
# for i in range(len(X_res)):
#     imgs.append(cv2.cvtColor(cv2.imread(X_res[i][0]), cv2.COLOR_BGR2GRAY))
#     print(X_res[i][0])

# print(imgs[0].shape)

imgs=[]
for i in range(len(X_res)):
    imgs.append(cv2.imread(X_res[i][0]))
    print(X_res[i][0])

print(imgs[0].shape)  

imagenes_resize=[]
for imagen in range(len(imgs)):
    imagenes_resize.append(cv2.resize(imgs[imagen],(224,224)))

def roc_auc(y_true, y_pred):
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.float32)

# def create_cnn_model(optimizer='adam', filters=32, kernel_size=(3, 3), pool_size=(2, 2)):
#     model = Sequential()
#     model.add(Conv2D(filters, kernel_size, activation='relu', input_shape=(441, 562, 1)))
#     model.add(MaxPooling2D(pool_size=pool_size))
#     model.add(Flatten())
#     model.add(Dense(32, activation='relu'))
#    # model.add(Dense(1, activation='sigmoid'))
#     model.add(Dense(2, activation='softmax'))
#     model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#     return model
from keras.models import Model
def create_optimizer(optimizer, learning_rate):
    if optimizer == 'adam':
        return Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        return RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Optimizer {optimizer} not supported.")

def create_cnn_model(optimizer='adam', filters=32, kernel_size=(3, 3), pool_size=(2, 2), learning_rate=0.001, dropout_rate=0.5, dense_units=128):
    model = Sequential()

    # Capa de convolución con 'filters' filtros, cada uno de tamaño 'kernel_size', activación ReLU
    model.add(Conv2D(filters, kernel_size, input_shape=(7, 9, 1), activation='relu'))
    model.add(BatchNormalization())  # Capa de Batch Normalization

    # Capa de pooling para reducir la dimensionalidad
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

    # Capa de salida con 2 neuronas (clasificación binaria)
    model.add(Dense(2, activation='softmax'))

    optimizer = create_optimizer(optimizer, learning_rate)
    # Compilar el modelo con una tasa de aprendizaje personalizable
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

#Resumen del modelo
def build_model(backbone, lr=5e-4):
    # Obtén la salida de las capas intermedias de ResNet50
    for layer in backbone.layers:
        layer.trainable = False

    # Obtiene la capa específica del backbone que deseas utilizar para Grad-CAM
    specific_layer = backbone.get_layer('conv5_block3_3_conv')  # Reemplaza 'nombre_capa' con el nombre real de la capa

    # Crea un nuevo modelo solo con la capa específica del backbone y las capas adicionales
    specific_layer = backbone.get_layer('conv5_block3_3_conv')  # Reemplaza 'conv5_block3_3_conv' con el nombre real de la capa
    specific_layer.trainable = True
    #output = specific_layer.output

    x = layers.GlobalAveragePooling2D()(backbone.output)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    output = layers.Dense(2, activation='softmax')(x)

    model = Model(inputs=backbone.input, outputs=output)

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=lr),
        metrics=['accuracy']
    )

    return model

K.clear_session()
gc.collect()

resnet = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

#Crea un nuevo modelo con la capa específica del backbone para Grad-CAM y las capas adicionales
model = build_model(resnet, lr=1e-3)
model.summary()


# model = KerasClassifier(build_fn=create_cnn_model, verbose=1)
# print(model.get_params().keys())


# param_grid = {
#     'filters': [8, 16, 32, 64],
#     'kernel_size': [(3, 3),(4,4), (5,5)],
#     'pool_size': [(2, 2), (3, 3)],
#     'optimizer': ['adam', 'rmsprop']
#     # 'epochs': np.arange(1, 50, 1)
# }

param_grid = {
    'filters': [8, 16, 32, 64],
    'kernel_size': [(3, 3), (4, 4), (5, 5)],
    'pool_size': [(2, 2), (3, 3)],
    'optimizer': ['adam', 'rmsprop'],
    'dense_units': [64, 128, 256],  # Prueba diferentes tamaños de capas densas
    'learning_rate': [0.001, 0.01, 0.1],  # Prueba diferentes tasas de aprendizaje iniciales
    'dropout_rate': [0.2, 0.4, 0.5]  # Prueba diferentes valores de dropout
}


learn_control = ReduceLROnPlateau(monitor='val_loss', patience=5,
                                  verbose=1,factor=0.2, min_lr=1e-7)

early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=15) 


X_train, X_test, Y_train, Y_test = train_test_split(imagenes_resize, y_res, stratify= y_res, test_size=0.2, random_state=0)
    # Dividir el conjunto de entrenamiento en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X_train, Y_train, stratify=Y_train,test_size=0.15, random_state=0)
    
X_train=np.array(X_train)
X_test=np.array(X_test)
X_val=np.array(X_val)

# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
# X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)

# # # X_test = X_test.reshape(X_test.shape[0], 7, 9, 1)
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

Y_train = to_categorical(y_train, num_classes= 2)
Y_test=to_categorical(Y_test, num_classes= 2)
Y_val=to_categorical(y_val, num_classes= 2)

train_generator = ImageDataGenerator(
        zoom_range=2, 
        rotation_range = 90,
        horizontal_flip=True, 
        vertical_flip=True, 
    )

batch_size=32
csv_file_path = str(Path.joinpath(consts.PATH_PROJECT_SAVE_MODEL, f'weights_best_{args.dataset}_batch_{batch_size}.hdf5'))

# #filepath="weights"+str(semilla)+ ".best.hdf5"
checkpoint = ModelCheckpoint(csv_file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

model = build_model(resnet, lr=1e-2)
    #model.summary()
history = model.fit_generator(
train_generator.flow(X_train, Y_train, batch_size=batch_size),
steps_per_epoch=X_train.shape[0] / batch_size,
epochs=200,
validation_data=(X_val, Y_val),
callbacks=[learn_control, early, checkpoint])

# hist_df = pd.DataFrame(history.history) 

# hist_csv_file = 'history_model_'+str(0)+'.csv'

# with open(hist_csv_file, mode='w') as f:
#     hist_df.to_csv(f)
# print(X_train.shape)
# Realiza la búsqueda de hiperparámetros
# grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
# grid_result = grid.fit(X_train, Y_train, steps_per_epoch=X_train.shape[0] / 8,epochs=200,validation_data=(X_val, Y_val),callbacks=[learn_control, early,checkpoint], generator=train_generator)


# # Muestra los resultados
# print("Mejor: %f usando %s" % (grid_result.best_score_, grid_result.best_params_))

# # Evalúa el modelo con los mejores hiperparámetros en un conjunto de prueba
# best_model = grid_result.best_estimator_
# accuracy = best_model.score(X_test, Y_test)
# print("Exactitud en conjunto de prueba: %f" % accuracy)


model.load_weights(csv_file_path)
accuracy=[]
recall=[]
especificidad=[]
auc=[]
Y_pred = model.predict(X_test)
Y_pred_train=model.predict(X_train)

Accuracy=accuracy_score(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1))
Recall=recall_score(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1))
tn, fp, fn, tp = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1)).ravel()
Especificidad = tn / (tn + fp)
fpr, tpr, threshold = metrics.roc_curve(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1))
roc_auc = metrics.auc(fpr, tpr)

accuracy.append(Accuracy)
recall.append(Recall)
especificidad.append(Especificidad)
auc.append(roc_auc)

print(accuracy, recall, especificidad, auc)