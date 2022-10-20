# %% [markdown]
# # EMOTION IA

# %% [markdown]
# ### DESCRIPCION DEL PROYECTO

# %% [markdown]
# La inteligencia artificial emocional o Emotion AI es una rama de la IA que permite que los ordenadores entiendan el lenguaje no verval
# humano como las posturas corporales o expresiones faciales.
# 
# * El objetivo de este proyecto es clasificar las emociones de las personas en funcion de sus imagenes faciales.
# * En este caso practico, asumiremos que trabajamos como consultores de IA / ML.
# * Una empresa emergente de San Diego nos ha contratado para construir, entrenar e implementar un sistema que monitoriza automaticamente
# las emociones y expresiones de las personas.
# * El equipo ha recopilado mas de 20.000 imagenes faciales, con sus etiquetas de expresion facial asociadas y alrededor de 2.000 imagenes
# con sus anotaciones faciales de puntos clave.
# 

# %% [markdown]
# ### PARTE 1: DETECCION DE PUNTOS FACIALES CLAVES
# 
# * En la parte 1, crearemos un modelo de aprendizaje profundo (Deep Learning) basado e la red neuronal convolucional y los bloques
# residuales para predecir puntos claves faciales.
# * El connjunto de datos consta de coordenadas x e y de 15 puntos clave faciales.
# * Las imagenes de entrada son de 96*96 pixeles.
# * Las imagenes constan de un solo canal de color (imagenes en escala de grises)

# %% [markdown]
# ###  LIBRERIAS

# %%
import pandas as pd
import numpy as np
import os
import PIL
import seaborn as sns
import pickle
from PIL import *
import cv2
import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from IPython.display import display
#from tensorflow.python.keras import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
#from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
import keras

# %%
print("Pandas Versio: %s" % pd.__version__)
print("Numpy Version: %s" % np.__version__)
print("PIL Version: %s" % PIL.__version__)
print("OpenCV Version: %s" % cv2.__version__)
print("TensorFlow Version: %s" % tf.__version__)
print("Keras Version: %s" % keras.__version__)
print("Sklearn Version: %s" % sklearn.__version__)

# %% [markdown]
# ### IMPORTAMOS EL DATASET

# %%
path_data = "../dataset/EmotionIA/"

# %%
keyfacial_df = pd.read_csv(path_data + 'data.csv')

# %%
print("Shape of keyfacial_df : %s" % str(keyfacial_df.shape))
print("\n")
keyfacial_df.head()

# %%
# Obtenemos informacion relevante del dataset
keyfacial_df.info()

# %%
# Comprobamos si hay valores nulos en el dataset
keyfacial_df.isnull().sum()

# %%
# Chequeamos el tamaño de la imagen con la ultima columna del dataset Image
keyfacial_df['Image'].shape

# %%
""" 
Dado que los valores para la imagen se dan como cadenas separadas por espacios, 
separamos los valores usando '' como separador.
Luego convertimos esto en una matriz numerica usando np.fromstring 
y convierta la matriz ID obtenida en una matriz 2D de forma (96, 96)
"""
keyfacial_df['Image'] = keyfacial_df['Image'].apply(
    lambda x: np.fromstring(x, dtype = int, sep = ' ').reshape(96, 96))

# %%
# Obtener la dimensiones de la imagen 0
keyfacial_df['Image'][0].shape

# %% [markdown]
# ### TAREA 1 
# * Obtener los valores promedio, minimo y maximo para `right_eye_center_x`.

# %%
min_rigth_eye_center_x = np.min(keyfacial_df['right_eye_center_x'])
max_rigth_eye_center_x = np.max(keyfacial_df['right_eye_center_x'])
mean_right_eye_center_x = np.mean(keyfacial_df['right_eye_center_x'])

print("rigth_eye_center_x")
print("Minimo: %s" % str(min_rigth_eye_center_x))
print("Maximo: %s" % str(max_rigth_eye_center_x))
print("Promedio: %s" % str(mean_right_eye_center_x))

# %%
keyfacial_df.describe()

# %% [markdown]
# ### VISUALIZACION DE IMAGENES

# %%
"""
Representamos una imagen aleatoria del conjunto de datos 
con puntos claves faciales.
Los datos de la imagen se obtienen del data['Image'] y 
se representan usando plt.imshow()
15 Coordenadas x e y para la imagen correspondiente.
Dado que las coordenadas x estan en columnas pares como 0,2,...
y las coordenadas y estan en columnas impares como 1,3,...
Accedemos a su valor usando el comando .loc que obtiene los valores
de las coordenadas de la imagen en funcion de la columna a la que refiere
"""

i = np.random.randint(1, len(keyfacial_df))
plt.imshow(keyfacial_df['Image'][i], cmap = 'gray')
print(i)
for j in range(1, 31, 2):
    plt.plot(keyfacial_df.loc[i][j-1], keyfacial_df.loc[i][j], 'rx')

# %%
# Veamos mas imagenes en formato matricial
fig = plt.figure(figsize = (20, 20))

for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1)
    plt.imshow(keyfacial_df['Image'][i], cmap = 'gray')
    for j in range(1, 31, 2):
        plt.plot(keyfacial_df.loc[i][j-1], keyfacial_df.loc[i][j], 'rx')

# %% [markdown]
# ### TAREA 2
# * Realiza una verificacion adicional en los datos visualizando aleatoriamente 64 imagenes junto con sus puntos clave correspondientes

# %%
# Solucion TAREA 2
fig = plt.figure(figsize = (20, 20))

list_images = []
for i in range(64):
    k = np.random.randint(1, len(keyfacial_df))

    while list_images.count(k) > 0: 
        k = np.random.randint(1, len(keyfacial_df))
    
    list_images.append(k)
    ax = fig.add_subplot(8, 8, i + 1)    
    plt.imshow(keyfacial_df['Image'][k], cmap = 'gray')
    for j in range(1, 31, 2):
        plt.plot(keyfacial_df.loc[k][j-1], keyfacial_df.loc[k][j], 'rx')

print(list_images)

# %% [markdown]
# ### AUMENTACION DE IMAGENES

# %% [markdown]
# La aumentacion de imagenes es para crear un dataset adicional, que en lugar de que utilize unicamente los datos del dataset
# de entrada que hemos cargados. Utiliza los datos del dataset de entrada, vamos a crear un nuevo dataset tomando las imagenes que tiene el dataset de entrada, vamos a girar las imagenes, incrementar el brillo o reducir el brillo, zoom. Haciendo esto vamos a conseguir algo muy importante en las redes
# neuronales que trabajan con imagenes, vamos a mejorar la generalizacion, la compatilibiladad de que el modelo trabaje bien, que es una caracteristica
# ensencial en los modelos de inteligencia artificial.

# %%
# Creamos una copia del dataframe
import copy

keyfacial_df_copy = copy.copy(keyfacial_df)

# Obtenemos las columnas del dataframe
columns = keyfacial_df_copy.columns[:-1]
columns

# %%
# Horizontal Flip - Damos vuelta a las imagenes en entorno al eje y
keyfacial_df_copy['Image'] = keyfacial_df_copy['Image'].apply(
    lambda x: np.flip(x, axis = 1))

# Dado que estamos volteando horizontalmente. los valores de la 
# coordenada y serian los mismos
# Solo cambiarian los valores de la coordenada x, todo lo que tenemos
# que hacer es restar nuestros valores iniciales de la coordenada x
# del ancho de la imagen(96)
for i in range(len(columns)):
    if i%2 == 0:
        keyfacial_df_copy[columns[i]] = keyfacial_df_copy[columns[i]].apply(
            lambda x: 96. - float(x))

# %%
# Mostramos la imagen original
plt.imshow(keyfacial_df['Image'][0], cmap = 'gray')

for j in range(1, 31, 2):
        plt.plot(keyfacial_df.loc[0][j-1], keyfacial_df.loc[0][j], 'rx')


# %%
# Mostramos la imagen girada horizontalmente
plt.imshow(keyfacial_df_copy['Image'][0], cmap = 'gray')

for j in range(1, 31, 2):
        plt.plot(keyfacial_df_copy.loc[0][j-1], keyfacial_df_copy.loc[0][j], 'rx')

# %%
# Concatenamos el dataset original con el dataframe aumentado
augmented_df = np.concatenate((keyfacial_df, keyfacial_df_copy))
augmented_df.shape

# %%
# Aumentar aleatoriamente el brillo de las imagenes
# Multiplicamos los valores de los pixeles por valores aleatorios entre 1,5 y 2
# para aumentar el brillo de la imagen
# Recortamos el valor entre 0 y 255

import random

keyfacial_df_copy = copy.copy(keyfacial_df)
keyfacial_df_copy['Image'] = keyfacial_df_copy['Image'].apply(
    lambda x: np.clip(random.uniform(1.5, 2) * x, 0.0, 255.0))

augmented_df = np.concatenate((augmented_df, keyfacial_df_copy))
augmented_df.shape

# %%
# Mostramos la imagen con aumento de brillo 
plt.imshow(keyfacial_df_copy['Image'][0], cmap = 'gray')

for j in range(1, 31, 2):
        plt.plot(keyfacial_df_copy.loc[0][j-1], keyfacial_df_copy.loc[0][j], 'rx')

# %% [markdown]
# ### TAREA 3
# * Aumenta las imagenes volteandolas verticalmente (Sugerencia: voltea a lo largo del eje x y teng en cuenta que si lo hacemos a lo largo
# del eje x las coordenadas x no cambiaran)

# %% [markdown]
# ### TAREA 4
# * Visualiza los resultados

# %%
keyfacial_df_copy = copy.copy(keyfacial_df)

# Obtenemos las columnas del dataframe
columns = keyfacial_df_copy.columns[:-1]

# Vertical Flip - Damos vuelta a las imagenes en entorno al eje x
keyfacial_df_copy['Image'] = keyfacial_df_copy['Image'].apply(
    lambda x: np.flip(x, axis = 0))

# Dado que estamos volteando verticalmente. los valores de la 
# coordenada x serian los mismos
# Solo cambiarian los valores de la coordenada y, todo lo que tenemos
# que hacer es restar nuestros valores iniciales de la coordenada y
# del ancho de la imagen(96)
for i in range(len(columns)):
    if i%2 == 1:
        keyfacial_df_copy[columns[i]] = keyfacial_df_copy[columns[i]].apply(
            lambda x: 96. - float(x))

# Mostramos la imagen girada verticalmente
plt.imshow(keyfacial_df_copy['Image'][0], cmap = 'gray')

for j in range(1, 31, 2):
        plt.plot(keyfacial_df_copy.loc[0][j-1], keyfacial_df_copy.loc[0][j], 'rx')

# Concatenamos el dataset
augmented_df = np.concatenate((augmented_df, keyfacial_df_copy))
augmented_df.shape

# %% [markdown]
# ### NORMALIZACION DE LOS DATOS Y PREPARACION PARA EL ENTRENAMIENTO

# %%
"""
Obtenemos el valor de las imagenes que esta presente en la columna 31 (dado que
el indice comienza desde 0, nos referimos a la columna 31 por 30 en python)
"""
img = augmented_df[:,30]

# Normalizamos las imagenes
img = img / 255.

# Creamos un array vacio de tamaño (x,96,96,1) para sumunistrar al modelo
X = np.empty((len(img), 96, 96, 1))

# Iteramos sobre la lista de imagenes y añadimos las nusnas ak array vacio
# tras expandir su dimencion de (96,96) a (96,96,1)
for i in range(len(img)):
    X[i,] = np.expand_dims(img[i], axis = 2)

# Convertimos el tipo array a float32
X = np.asarray(X).astype(np.float32)
X.shape

# %%
# Obtenemos el valor de las coordenadas x & y que se utilizaran como tarjet
y = augmented_df[:, :30]
y = np.asarray(y).astype(np.float32)
y.shape

# %%
# Dividimos los datos en entrenamiento y testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# %% [markdown]
# ### TAREA 5
# * Intentar usar un valor diferente para `test_size` y verifica que la division es correcta

# %%
print("x_train shape: %s" % str(x_train.shape))
print("x_test shape: %s" % str(x_test.shape))

# %% [markdown]
# ### ENTENDER LA TEORIA E INTUICION DE LAS REDES NEURONALES
# 
# * El cerebro tiene mas de 100 mil millones de neuronas que se comunican a atraves de señales electricas y quimicas.
# Las neuronas se comunican entre si y nos ayudan a ver, pensar, etc
# * El cerebro humano aprende creando conexiones entre estas neuronas. Las RNA son modelo de procesamiento de informacion
# inspirados en el cerebro humano
# * La neurona recolecta señales de los canales de entradas llamados dentritas, procesa la infomacion en su nucleo y 
# luego genera una salida en una rama larga y delgada llamada axon
# * El sesgo o bias permite cambiar la curva de la funcion de activacion hacia arriba o abajo.
# * Numero de parametros ajustables = 4 (3 pesos y 1 bias)
# `y=f(x1.w1 + x2.w2 + x3.w3 + b)` las Xi son los inputs y los Wi son los pesos.

# %% [markdown]
# ### TAREA 6
# * Lista al menos 3 funciones diferentes de activacion e indica cual es el tipo preferido para usar en las capas ocultas
# 
# * 1- Funcion sigmoide -> toma numberos y devuelve un valor entre 0 y 1, convierte numeros negativos muy pequeños a 0 y numeros positivos muy grandes a 1, 
# generalmente se usa en la capa de salida (output layer)
# * 2- Funcion Relu -> si el input x < 0, la salida es 0 y si x > 0 la salida es x.
# Es usada en limites simples debido a su eficiencia computacional, se usa generalmente en las capas ocultas (hidden layer).
# * 3- Funcion Tangente Hiperbolica -> es similar a la funcion sigmoide pero el rango de valores es de -1 a 1

# %% [markdown]
# #### RED PERCEPTRON DE MULTIPLES CAPAS
# 
# * Conectemos varias de esta neuronas de forma multicapa.
# * Cuantas mas capas ocultas, mas profunda se volvera la red

# %% [markdown]
# ### TAREA 7
# * Lista almenos 3 redes neuronales diferentes y sus respectivas aplicaciones.
# 
# ##### Resolucion
# - Red Neuronal Artificial -> Una sola neurona puede verse como una regresion logistica, pero la Red Neuronal Artificial es un grupo de multiples neuronas
# en cada capa, esta red neuronal se conoce tambien como Feed-Forward Neural Network ya que la entradas solo se procesan en la direccion de avance. Esta se divide en 3 capas, la capa de entrada acepta los inputs, la capa oculta procesa estos inputs y la capa de salida produce el resultado. La RNA es capaz de aprender cualquier funcion no lineal gracias a las funciones de activacion, estas introducen propiedades no lineales a la red y esto ayuda a aprender cualquier relacion completa entre la entrada y salida.
# 
# - Red Neuronal Recurrente -> A diferencia de la RNA, la red recurrente tiene una conexion recurrente en la capa oculta, esta restriccion de bucle garantiza que la informacion secuencial se capture en los datos de entrada.
# 
# - Red Neuronal de Convolucion -> Estas redes estan de moda en la comunidad de Deep Learning en la actualidad. Estos modelos de RNC se utilizan en diferentes aplicaciones y son muy frecuentes en proyecto de procesamiento de imagenes y videos. Los componentes principales de las CNN son filtros o conocidos como nucleos o convoluciones. Estas convoluciones se utilzan para extraer caracteristicas relevantes de la entrada mediante la operacion de Convolucion. Convolucionar una imagen da como resultado un mapa de caracteristicas o mejor dicho un feature_map

# %% [markdown]
# ### ENTENDER EL ENTRENAMIENTO EN REDES Y LOS ALGORITMOS DE GRADIENTE DESCENDENTE

# %% [markdown]
# #### Division de datos en training y testing
# 
# * El conjunto de datos generalmente se divide en 80% para entrenamiento y 20% para pruebas.
# * A veces, tambien podemos incluir un conjunto de datos validacion cruzada y luego lo dividimos en segmentos de 60%, 20%, 20% para entrenamiento, validacion y prueba respectivamente.
#     * Conjunto de entrenamiento: se utiliza para calcular el gradiente y actualizar los pesos de la red.
#     * Conjunto de validacion: utilizado para la validacion cruzada para evaluar la calidad del entrenamiento. La validacion cruzada se implementa para superar el overfitting que se produce cuando el algoritmo se centra en los detalles del conjunto de entrenamiento a costa de perder capacidad de generalizacion.
#     * Conjunto de pruebas: usado para probar la red entrenada
# 
# #### Gradiente Descendente
# 
# * El gradiente descendente es un algoritmo de optimizacion que se utiliza para obtener el peso de red optimizado y los valores del bias.
# * Funciona intentando minimizar de forma iterativa la funcion de coste.
# * Funciona calculando el gradiente de la funcion de costes y moviendose en la direccion negativa hasta que se alcanza el minimo local / global.
# * Si se toma el valor positivo del gradiente, se alcanza el maximo local / global.
# * El tamaño de los pasos dados a cada iteracion se llama tasa de aprendisaje.
# * Si la tasa de aprendizaje aumenta, el area cubierta en el espacio de busqueda aumentara para que podamos alcanzar el minimo global mas rapido.
# * Para tasas de aprendizaje pequeñas, el entrenamiento llevara mucho mas tiempo para alcanzar valores de peso optimizado.
# 
# #### TAREA 8:
# * Que ocurre cuando configuramos el ratio de aprendizaje (learning rate) a los valores extremos (valores muy pequeños y valores muy grandes)?
# Como se puede conseguir el mejor resultado de ambos escenarios?
# 
# `Un learning rate con un valor demasiado alto implica que el modelo converge de forma muy rapida obteniendo un resultado no optimo mientras que un learning rate con un valor demasiado pequeño hace que el proceso de convergencia tarde mas tiempo. Por lo cual una solucion para este problema es el Gradiente Descendente Estocasico que basicamente se trata de un learning rate dinamico de modo que se empieza con un valor original para el learning rate, empezariamos con un valor alto al principio y luego seria dinamico multiplicando por un valor en cada iteracion haciendo que el learning rate sea mas pequeño en cada iteracion`
# 

# %% [markdown]
# ### ENTENDER LA TEORIA E INTUICION DETRAS DE LAS REDES NEURONALES CONVOLUCIONALES Y RESNETS

# %% [markdown]
# ### RESNET (REDES RESIDUALES)
# 
# * A medida que las redes neuronales convolucionales (RNC) se hacen mas profundas, tienden a ocurrir el desvanecimiento del gradiente que impacta negativamente en el rendimiento de la red
# * El problema del desvanecimiento del gradiente ocurre cuando el gradiente se propaga hacia atras a capas anteriores, lo que da como resultado un gradiente muy pequeño
# * La red neuronal residual incluye la funcion de "omision sin conexion" que permite el entrenamiento de 152 capas sin el problema del desvanecimiento del gradiente
# * ResNet funciona agregando "asignaciones de identidad" en la parte superior de la RNC
# * ImageNet contiene 11 millones de imagenes y 11.000 categorias
# * ImageNet se utiliza para entrenar la red profunda de ResNet
# 
# ### TAREA 9
# * ¿Cual es la eficacia de las ResNets comparada con AlexNet en datasets de ImageNet?
# 
# La eficacia de ResNet sobre AlexNet radica en la profundidad de las redes ya que AlexNet tiene una profundidad de 8 capas ocultas convolucionales y ResNet tiene una profundidad de 152 capas ocultas convolucionales. Sin embargo aumentar la profundidad de una red neuronal no es solo apilar capas una detras de otra ya que estas redes son dificiles de entrenar debido al desvanecimiento del gradiente, ya que el gradiente se propaga hacia atras  a las capas anteriores, dado que la multiplicacion repetida puede hacer que el valor del gradiente sea muy pequeño y como resultado provoca que la red se sature o que comienze a degradarse rapidamente.
# ##### Omitir conexion: la fuerza de ResNet
# Para resolver el problema del gradiente de fuga / explosión, esta arquitectura introdujo el concepto llamado Red residual. En esta red usamos una técnica llamada salto de conexiones . La conexión de salto omite el entrenamiento de algunas capas y se conecta directamente a la salida.
# 
# Estas son las ventajas de ResNet sobre AlexNet

# %% [markdown]
# ### CONSTRUIR UN MODELO DE RED NEURONAL RESIDUAL PROFUNDA PARA CREAR UN MODELO QUE DETECTE PUNTOS FACIALES CLAVE

# %%
def residual_block(X, filter, stage):
    # Bloque Convolucional
    X_copy = X 
    
    f1, f2, f3 = filter

    # Camino Principal
    X = Conv2D(f1, kernel_size = (1, 1), strides = (1, 1), name = 'res_' + str(stage) + '_conv_a', 
        kernel_initializer = glorot_uniform(seed = 0))(X)
    X = MaxPool2D((2, 2))(X)
    X = BatchNormalization(axis = 3, name = 'bn' + str(stage) + '_conv_a')(X)
    X = Activation('relu')(X)

    X = Conv2D(f2, kernel_size = (3, 3), strides = (1, 1), padding = 'same', 
        name = 'res_' + str(stage) + '_conv_b', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = 'bn' + str(stage) + '_conv_b')(X)
    X = Activation('relu')(X)

    X = Conv2D(f3, kernel_size = (1, 1), strides = (1, 1), name = 'res_' + str(stage) + '_conv_c',
        kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = 'bn' + str(stage) + '_conv_c')(X)

    # Camino Corto
    X_copy = Conv2D(f3, kernel_size = (1, 1), strides = (1, 1), name = 'res_' + str(stage) + '_conv_copy', 
        kernel_initializer = glorot_uniform(seed = 0))(X_copy)
    X_copy = MaxPool2D((2, 2))(X_copy)
    X_copy = BatchNormalization(axis = 3, name = 'bn' + str(stage) + '_conv_copy')(X_copy)

    # Añadir 
    X = Add()([X, X_copy])
    X = Activation('relu')(X)

    # Bloque de Identidad 1
    X_copy = X

    # Camino Principal
    X = Conv2D(f1, (1,1),strides = (1,1), name = 'res_' + str(stage) + '_identity_1_a',
        kernel_initializer= glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis =3, name = 'bn_' + str(stage) + '_identity_1_a')(X)
    X = Activation('relu')(X) 

    X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name = 'res_' + str(stage) + '_identity_1_b', 
        kernel_initializer= glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis =3, name = 'bn_' + str(stage) + '_identity_1_b')(X)
    X = Activation('relu')(X) 

    X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name = 'res_' + str(stage) + '_identity_1_c', 
        kernel_initializer= glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis =3, name = 'bn_' + str(stage) + '_identity_1_c')(X)

    # Añadir
    X = Add()([X,X_copy])
    X = Activation('relu')(X)

    # Bloque de Identidad 2
    X_copy = X


    # Camino Principal
    X = Conv2D(f1, (1,1),strides = (1,1), name = 'res_'+str(stage) + '_identity_2_a',
        kernel_initializer= glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_a')(X)
    X = Activation('relu')(X) 

    X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name = 'res_' + str(stage) + '_identity_2_b',
        kernel_initializer= glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis =3, name = 'bn_' + str(stage) + '_identity_2_b')(X)
    X = Activation('relu')(X) 

    X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name = 'res_' + str(stage) + '_identity_2_c',
        kernel_initializer= glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis =3, name = 'bn_' + str(stage) + '_identity_2_c')(X)

    # Añadir
    X = Add()([X,X_copy])
    X = Activation('relu')(X)

    return X

# %%
input_shape = (96, 96, 1)

# Tamaño del tensor de entrada
X_input = Input(input_shape)

# Zero Padding 
X = ZeroPadding2D((3, 3))(X_input)

# 1 - Fase
X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', 
    kernel_initializer = glorot_uniform(seed = 0))(X)
X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
X = Activation('relu')(X)
X = MaxPooling2D((3, 3), strides = (2, 2))(X)

# 2 - Fase
X = residual_block(X, filter = [64, 64, 256], stage = 2)

# 3 - Fase 
X = residual_block(X, filter = [128, 128, 512], stage = 3)

# 4 - Fase
#X = residual_block(X, filter = [256, 256, 1024], stage = 4)

# Average Pooling
X = AveragePooling2D((2, 2), name = 'Average_Pooling')(X)

# Capa Final
X = Flatten()(X)
X = Dense(4096, activation = 'relu')(X)
X = Dropout(0.2)(X)
X = Dense(2048, activation = 'relu')(X)
X = Dropout(0.1)(X)
X = Dense(30, activation = 'relu')(X)

model_1_facialKeyPoints = Model(inputs = X_input, outputs = X)
model_1_facialKeyPoints.summary()

# %% [markdown]
# ### TAREA 10 
# * Experimenta cambiando la arquitectura de la red eliminando 2 capas MaxPooling del Bloque Res y entrena el modelo de nuevo
# * Intenta agregar el bloque `X = res_block(X, filter[256,256,1024], stage = 4)` despues del bloque de la etapa #3 
# * Que observas? Comenta tu respuesta
# 
# 1 - Al eliminar las capas de MaxPooling del Bloque Res lo que sucede es que aumenta de forma exponencial los parametros del modelo
# 
# 2 - Al agregar el bloque residual aumenta el tamaño de parametros del modelo y tambien la cantidad de parametros no entrenable

# %% [markdown]
# ### COMPILAR Y ENTRENAR EL MODELO DE DEEP LEARNING PARA LA DETECCION DE PUNTOS FACIALES

# %%
# Ver temas de versiones de las librerias
#adam = tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0.9, 
#    beta_2 = 0.999, amsgrad = False, name = 'Adam')

#model_1_facialKeyPoints.compile(loss = 'mean_squared_error', optimizer = adam, 
#    metrics = ['accuracy'])

model_1_facialKeyPoints.compile(loss = 'mean_squared_error', optimizer = 'adam',  
    metrics = ['accuracy'])

# Guardamos el mejor modelo con mejor error de validación 
checkpointer = ModelCheckpoint(filepath = "../models/emotion_ia/FacialKeyPoints_weights.hdf4", verbose = 1, save_best_only = True)

# %%
# Creamos el historial del entrenamiento
history = model_1_facialKeyPoints.fit(x_train, y_train, batch_size = 32,
    epochs = 40, validation_split = 0.05, callbacks = [checkpointer])

# %%
model_json = model_1_facialKeyPoints.to_json()
with open('../models/emotion_ia/json/FacialKeyPoints-model.json', 'w') as json_file:
    json_file.write(model_json)

# %% [markdown]
# ### EVALUAR LA EFICACIA DEL MODELO DE DETECCION DE PUNTOS FACIALES CLAVES ENTRENADO

# %%
with open('../models/emotion_ia/json/FacialKeyPoints-model.json', 'r') as json_file:
    json_saveModel = json_file.read()

# Cargar la arquitectura del modelo
model_1_facialKeyPoints = tf.keras.models.model_from_json(json_saveModel)
model_1_facialKeyPoints.load_weights('../models/emotion_ia/FacialKeyPoints_weights.hdf4')
model_1_facialKeyPoints.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])

# %%
# Evaluar el modelo
result = model_1_facialKeyPoints.evaluate(x_test, y_test)
print("Accuracy : {}".format(result[1]))

# %%
# Obetenemos las claves del modelo
history.history.keys()

# %%
# Representamos los scores del entrenamiento
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'], loc = 'upper right')
plt.show()

# %%
# Representamos los scores del entrenamiento
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_accuracy', 'val_accuracy'], loc = 'lower right')
plt.show()

# %% [markdown]
# ## PARTE 2 - DETECCION DE LAS EXPRESIONES FACIALES

# %% [markdown]
# * El segundo modelo clasificara las emociones de las personas
# * El dataset contengra imagenes de las siguientes categorias
#     - 0 = Enfadado
#     - 1 = Disgustado
#     - 2 = Triste
#     - 3 = Feliz
#     - 4 = Sorprendido

# %% [markdown]
# ### IMPORTAR & EXPLORAR EL DATASET PARA LA DETECCION DE EXPRESIONES FACIALES

# %%
# Leemos el CSV
facialExpression_df = pd.read_csv(path_data + 'icml_face_data.csv')
print("Shape of facialExpression_df : {}".format(facialExpression_df.shape))
facialExpression_df.head()

# %%
# Vemos la primer imagen en pixeles
facialExpression_df[' pixels'][0] # Formato String

# %% [markdown]
# Los pixeles en formato string no nos sirve para lo que necesitamos, de modo que necesitamos crear 2 funciones, una que convierta el string a array y otra para redimensionar la imagen de (48,48) a (96,96) 

# %%
# Funcion para convertir valores de pixel de formato string a formato array
def stringToArray(x):
    return np.array(x.split(' ')).reshape(48, 48, 1).astype('float32')

# Redimensionamos la imagen de (48, 48) a (96, 96)
def resize(x):
    img = x.reshape(48, 48)
    return cv2.resize(img, dsize = (96, 96), interpolation = cv2.INTER_CUBIC)    

# %%
facialExpression_df[' pixels'] = facialExpression_df[' pixels'].apply(lambda x: stringToArray(x))
facialExpression_df[' pixels'] = facialExpression_df[' pixels'].apply(lambda x: resize(x))

# %%
# Comprobamos la estructura del data frame
facialExpression_df.shape

# %%
facialExpression_df.head()

# %%
# Revisar si hay valores nulo en el data frame
facialExpression_df.isnull().sum()

# %%
label_to_text = {
    0: 'Ira',
    1: 'Disgusto',
    2: 'Tristeza',
    3: 'Felicidad',
    4: 'Sorpresa'
}

# %% [markdown]
# ### TAREA 12
# * Visualizar la primera imagen del dataframe y ver que la imagen no se distorciona al cambiar el tamaño o debido a las operaciones de remodelacion

# %%
plt.imshow(facialExpression_df[' pixels'][0], cmap = 'gray')

# %% [markdown]
# ### VISUALIZACION DE LAS IMAGENES Y CORRESPONDIENTE ETIQUETA

# %%
emotions = [0, 1, 2, 3, 4]

for i in emotions: 
    data = facialExpression_df[facialExpression_df['emotion'] == i][:1]
    img = data[' pixels'].item()
    img = img.reshape(96, 96)
    plt.figure()
    plt.title(label_to_text[i])
    plt.imshow(img, cmap = 'gray')

# %% [markdown]
# ### TAREA 13
# * Representar un grafico de barras para averiguar cuantas imagenes estan presentes por cada emocion

# %%
facialExpression_df.emotion.value_counts().index

# %%
facialExpression_df.emotion.value_counts()

# %%
plt.figure(figsize = (10, 10))
sns.barplot(x = facialExpression_df.emotion.value_counts().index,
    y = facialExpression_df.emotion.value_counts())

# %% [markdown]
# ### PREPARACION DE LOS DATOS Y AUMENTACION DE LAS IMAGENES

# %%
from keras.utils import to_categorical

X = facialExpression_df[' pixels']
y = to_categorical(facialExpression_df['emotion'])

# %%
X[0]

# %%
y

# %%
X.shape[0]

# %%
# El metodo stack apila toda la informacion en el axis = 0
X = np.stack(X, axis = 0)
X = X.reshape(X.shape[0], 96, 96, 1)

print("X Shape: {}".format(X.shape))
print("y Shape: {}".format(y.shape))

# %%
# Dividir el dataframe en conjunto de entrenamiento, test y validacion
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, shuffle = True)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size = 0.5, shuffle = True)

# %%
print("x_val Shape: {}".format(x_val.shape))
print("y_val Shape: {}".format(y_val.shape))

# %%
print("x_test Shape: {}".format(x_test.shape))
print("y_test Shape: {}".format(y_test.shape))

# %%
print("x_train Shape: {}".format(x_train.shape))
print("y_train Shape: {}".format(y_train.shape))

# %%
# Pre procesado de Imagenes
x_train = x_train / 255
x_test = x_test / 255
x_val = x_val / 255

# Miramos el x_train como los pixeles estan en la escala de 0 y 1 
# 0 = pixel de color negro
# 1 = pixel de color blanco
# 0 < p < 1 = pixel en escala de grises
x_train

# %%
# Fíjate que "Brightness_range"
# 1.0 no afecta al brillo de la imagen 
# números más pequeños que 1.0 oscurecen la imagen [0.5, 1.0]
# números más grandes que 1.0 iluminan la imagen [1.0, 1.5] 

# %%
train_datagen = ImageDataGenerator(
    rotation_range = 15, 
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.1,
    zoom_range = 0.1,
    horizontal_flip = True,
    vertical_flip = True,
    brightness_range = [1.1, 1.5],
    fill_mode = "nearest"
)

# %% [markdown]
# ### CONSTRUIR Y ENTRENAR MODELO DE DEEP LEARNING PARA LA CLASIFICACION DE EXPRESIONES FACIALES

# %%
input_shape = (96, 96, 1)

# Tamaño del tensor de entrada
X_input = Input(input_shape)

# Zero-padding
X = ZeroPadding2D((3, 3))(X_input)

# 1 - Fase
X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', 
    kernel_initializer = glorot_uniform(seed = 0))(X)
X = BatchNormalization(axis =3, name = 'bn_conv1')(X)
X = Activation('relu')(X)
X = MaxPooling2D((3, 3), strides = (2, 2))(X)

# 2 - Fase
X = residual_block(X, filter= [64, 64, 256], stage= 2)

# 3 - Fase
X = residual_block(X, filter= [128, 128, 512], stage= 3)

# 4 - Fase
X = res_block(X, filter= [256, 256, 1024], stage= 4)

# Average Pooling
X = AveragePooling2D((2, 2), name = 'Average_Pooling')(X)

# Capa Final
X = Flatten()(X)
X = Dense(5, activation = 'softmax', name = 'Dense_final', 
    kernel_initializer = glorot_uniform(seed = 0))(X)

model_2_emotion = Model( inputs= X_input, outputs = X, name = 'Resnet18')
model_2_emotion.summary()


# %%
# Entrenar la red
model_2_emotion.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', 
    metrics = ['accuracy'])

# %%
# Recordemos que el primer modelo de puntos faciales claves se guardo con:
# FacialKeyPoints_weights.hdf4 and FacialKeyPoints-model.json

# Usamos la parada temprana para salir del entrenamiento si el  error de validacion
# no decrece despues de cierto numero de epochs (paciencia)
earlystopping = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 10)

# Guardamos el mejor modelo con menor error de validacion
checkpointer = ModelCheckpoint(filepath = "../models/emotion_ia/FacialExpression_weights.hdf4", 
    verbose = 1, save_best_only = True)

# %%
history = model_2_emotion.fit(train_datagen.flow(x_train, y_train),
    batch_size = 64,
    validation_data = (x_val, y_val),
    steps_per_epoch = len(x_train) // 64,
    epochs = 40, callbacks = [checkpointer, earlystopping]    
)

# %%
# Guardar la arquitectura del modelo en un JSON 
model_json = model_2_emotion.to_json()

with open("../models/emotion_ia/json/FacialExpression-model.json", "w") as json_file:
    json_file.write(model_json)

# %% [markdown]
# ### TAREA 15
# * Experimentar con varios tamaños de lote, paciencia, optimizadores, y arquitectura de red para mejorar el rendimiento de la red

# %% [markdown]
# ### EVALUAR LA EFICACIA DEL MODELO CLASIFICADOR DE EXPRESIONES FACIALES ENTRENADO

# %%
with open('FacialExpression-model.json', 'r') as json_file:
    json_saveModel = json_file.read()

# Cargamos la arquitectura del modelo
model_2_emotion = tf.keras.models.model_from_json(json_saveModel)
model_2_emotion.load_weights('../models/emotion_ia/FacialExpression_weights.hdf4')
model_2_emotion.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# %%
score = model_2_emotion.evaluate(x_test, y_test)
print('Accuracy en fase de test: {}'.format(score[1]))

# %%
history.history.keys()

# %%
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# %%
epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label = 'Accuracy en Entrenamiento')
plt.plot(epochs, val_accuracy, 'b', label = 'Accuracy en Validacion')
plt.title('Accuracy')
plt.legend()

# %%
plt.plot(epochs, loss, 'bo', label = 'Loss en Entrenamiento')
plt.plot(epochs, val_loss, 'b', label = 'Loss en Validacion')
plt.title('Loss')
plt.legend()

# %%
predicted_classes = np.argmax(model_2_emotion.predict(x_test), axis = -1)
y_true = np.argmax(y_test, axis = -1)

y_true.shape

# %%
# Matriz de confusion
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, predicted_classes)
plt.figure(figsize = (10, 10))
sns.heatmap(cm, annot = True, cbar = False)

# %% [markdown]
# ### TAREA 16
# * Mostrar una matriz de 25 imagenes junto con su etiqueta predicha / verdadera
# * Mostrar el informe de clasificacion y analizar la precision y la recuperacion

# %%
L = 5
W = 5

fig, axes = plt.subplots(L, W, figsize = (24, 24))
axes = axes.ravel()

for i in np.arange(0, L*W):
    axes[i].imshow(x_test[i].reshape(96, 96), cmap = 'gray')
    axes[i].set_title('Prediccion = {}\nVerdadera = {}'.format(label_to_text[predicted_classes[i]], label_to_text[y_true[i]]))
    axes[i].axis('off')    # QUITA LOS EJES

plt.subplots_adjust(wspace = 1)

# %%
# Classification report
from sklearn.metrics import classification_report

print(classification_report(y_true, predicted_classes))

# %% [markdown]
# ### COMBINAR AMBOS MODELOS (1) DETECCION DE PUNTOS CLAVE FACIALES Y (2) DE EXPRESIONES FACIALES

# %%
def predict(x_test):
    # Hacemos la prediccion con el modelo de puntos clave
    df_predict = model_1_facialKeyPoints.predict(x_test)

    # Hacemos la prediccion con el modelo de emociones
    df_emotion = np.argmax(model_2_emotion.predict(x_test), axis = -1)

    # Redimensionamos el array de (856, ) a (856, 1)
    df_emotion = np.expand_dims(df_emotion, axis = 1)

    # Convertimos las predicciones en un dataframe 
    df_predict = pd.DataFrame(df_predict, columns = columns)

    # Añadimos la emocion al dataframe de predicciones
    df_predict['emotion'] = df_emotion

    return df_predict

# %%
df_predict = predict(x_test)
df_predict.head()

# %% [markdown]
# ### TAREA 17
# * Representamos una matriz de 16 imagenes junto con emocion predicha y sus puntos faciales

# %%
# Representamos las imágenes de test junto con los puntos clave y emociones

fig, axes = plt.subplots(4, 4, figsize = (24, 24))
axes = axes.ravel()

for i in range(16):

    axes[i].imshow(x_test[i].squeeze(),cmap='gray')
    axes[i].set_title('Prediccion = {}'.format(label_to_text[df_predict['emotion'][i]]))
    axes[i].axis('off')
    for j in range(1,31,2):
            axes[i].plot(df_predict.loc[i][j-1], df_predict.loc[i][j], 'rx')

# %% [markdown]
# ### DESPLEGAR LOS MODELOS ENTRENADOS

# %% [markdown]
# #### GUARDAR EL MODELO ENTRENADO
# 
# Despliegue del Modelo utilizando TENSORFLOW SERVING:
# 
# * Supongamos que ya tenemos entrenado nuestro modelo y esta generando buenos resultados en los datos de prueba
# * Ahora, queremos integrar nuestro modelo de Tensorflow entrenado en una aplicacion web e implementar el modelo en un entorno de nivel de produccion
# * El siguiente objetivo se puede obtener utilizando TensorFlow Serving
# * Con la ayuda de TensorFlow Serving podemos implementar facilmente nuevos algoritmos para hacer predicciones
# * Para publicar el modelo entrenado con TensorFlow Serving necesitamos guardar el modelo en el formato que sea adecuado para entregar usando TensorFlow Serving
# * El modelo tendra un numero de version y se guardara en un directorio estructurado
# * Una vez que se guarda el modelo ahora podemos usar TensorFlow Serving para comenzar a realizar solicitures de inferencia utilizando una version especifica de nuestro modelo entrenado "servible"
# 
# Ejecutar TENSORFLOW SERVING:
# 
# * Parametros importantes:
#     * rest_api_port: puerto que usaremos para las peticiones REST
#     * model_name: la URL que usaremos para las peticiones REST, se puede elegir cualquier nombre
#     * model_base_path: la ruta al directorio donde hemos guardado el modelo
# * REST es una reinterpretacion del protocolo HTTP donde los comandos http tienen un significado semantico
# 
# Hacer peticiones con TENSORFLOW SERVING:
# 
# * Para hacer predicciones usando TensorFlow Serving, necesitamos pasar las solicitudes de inferencia (datos de nuestra imagen) como un objeto JSON
# * Luego, usamos la libreria requests de Python para realizar una solicitud por POST al modelo implementado, pasando el objeto JSON que contiene las solicitudes de inferencia (datos de nuestra imagen)
# * Finalmente, obtenemos la prediccion de la solicitud por POST realizada al modelo implementado y luego usamos la funcion argmax para encontrar la clase predicha
# 
# RESUMEN:
# 
# * Ahora necesitamos guardar nuestro modelo entrenado y debe guardarse en un formato `SaveModel`
# * El modelo tendra un numero de version y se guardara en un directorio estructurado
# * `tf.saved_model.save` es una funcion que se usa para crear un modelo guardado que es adecuado para publicar con TensorFlow Serving
# * Una vez guardado el modelo, ahora podemos usar TensorFlow Serving para comenzar a realizar solicitudes de inferencia utilizando una version especifica de nuestro modelo entrenado "servible"
# * Utilizaremos `SavedModel` para guardar y cargar nuestro modelo: variables, el grafico y los metadatos del grafico
# * Mas Info: https://www.tensorflow.org/guide/saved_model

# %%
import json 
from tensorflow.keras import backend as K

def deploy(directory, model):
    MODEL_DIR = directory
    version = 1

    # Juntamos el directorio del temp model con la version elegida
    # El resultado sera = '\tmp\version_number'
    export_path = os.path.join(MODEL_DIR, str(version))
    print('export_path = {}\n'.format(export_path))

    # Guardemos el modelo con saved_model.save()
    # Si el directorio existe, debemos borrarlo con '!rm'
    # rm elimina cada fichero especificado usando la consola de comandos
    if os.path.isdir(export_path):
        print('\nAlready saved a model, cleaning up\n')
        !rm -r {export_path}

    tf.saved_model.save(model, export_path)

    os.environ["MODEL_DIR"] = MODEL_DIR

# %% [markdown]
# ### PUBLICAR EL MODELO CON TENSORFLOW SERVING

# %%
import sys

# We need sudo prefix if not on a Google Colab.
if 'google.colab' not in sys.modules:
  SUDO_IF_NEEDED = 'sudo'
else:
  SUDO_IF_NEEDED = ''

SUDO_IF_NEEDED

# %%
import getpass

password = getpass.getpass()
command = "echo 'deb http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal'"
os.popen(command, 'w').write(password+'\n')



