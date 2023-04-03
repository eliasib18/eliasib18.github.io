# Proyecto 1: Identificación de números por medio de imágenes

## Descripción
El trabajo consiste en identificar grupos de imágenes para reconocimiento de números. Para esto, se deberán realizar los siguientes pasos:
1. Dado que nuestros datos están en diferentes escalas, es necesario normalizar los datos.
2. Aplicar un método de reducción de dimensionalidad y visualizar los datos
3. Buscar grupos en los datos reducidos con alguna técnica de agrupamiento o clasificación.
4. Interpretar los resultados.
5. Dadas dos imágenes nuevas, identificar a que grupo pertenece. (Inferencia)


## A continuación se muestran las librerias utilizadas junto con una breve descripción de ellas.

### 1. numpy: 
Una librería de Python para realizar operaciones matemáticas en matrices y arreglos n-dimensionales.

### 2. matplotlib.pyplot: 
Una librería de Python para visualizar datos en gráficos y diagramas.

### 3. sklearn.linear_model: 
Un módulo de la librería scikit-learn para ajustar modelos de regresión lineal y logística.

### 4. sklearn.cluster: 
Un módulo de la librería scikit-learn para realizar clustering o agrupamiento de datos en grupos similares.

### 5. sklearn.decomposition: 
Un módulo de la librería scikit-learn para realizar técnicas de reducción de dimensionalidad como PCA (Análisis de Componentes Principales).

### 6. sklearn.manifold: 
Un módulo de la librería scikit-learn para reducción de dimensionalidad no lineal como t-SNE (t-Distributed Stochastic Neighbor Embedding).

### 7. sklearn.preprocessing: 
Un módulo de la librería scikit-learn para realizar transformaciones en los datos antes de aplicar modelos de aprendizaje automático, como la estandarización de características.

### 8. sklearn.datasets: 
Un módulo de la librería scikit-learn que incluye conjuntos de datos de ejemplo para practicar el aprendizaje automático.

### 9. sklearn.model_selection:
Un módulo de la librería scikit-learn para realizar evaluación de modelos, como la división de los datos en entrenamiento y prueba, y la validación cruzada.

### 10. sklearn.metrics:
Un módulo de la librería scikit-learn para calcular métricas de evaluación de modelos, como la precisión, el recall y la F1-score.

### 11. warnings: 
Un módulo de Python para gestionar advertencias y mensajes de aviso durante la ejecución del código. En este caso, se utiliza para desactivar las advertencias en la salida del código.

#### -------------------------------------------------------------------------------------------------------------------------------------------------------------
 Asi mismo, en este proyecto se recomienda el uso de las funciones y clases integradas de scikit-learn. Para entender el uso de estas clases y ver algunos ejemplos puedes consultar la documentación oficial

- [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE)
- [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [DBScan](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

## Nota: Existen múltiples soluciones a este problema, por lo que la solución que mostraremos no es la única.

Empezamos bien con la importación de las librerias a utilizar:


```
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.cluster import KMeans, DBSCAN

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split

from sklearn import metrics

import warnings

warnings.filterwarnings('ignore')

#Seed

np.random.seed(202)
```

## 1. Analizando los datos 

Comenzamos leyendo nuestros datos y visualizando algunos ejemplos para analizarlos. En este caso utilizaremos el [digits dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html#sphx-glr-auto-examples-datasets-plot-digits-last-image-py). En este dataset encontrarás 1797 imágenes de 8x8. Cada imagen es un dígito escrito a mano. Primero separaremos los datos en entrenamiento y validación

A continuación el siguiente código carga el conjunto de datos usando la función load_digits del módulo sklearn.datasets. Luego, divide los datos en un conjunto de entrenamiento y un conjunto de validación utilizando la función train_test_split del módulo sklearn.model_selection. La división se realiza de forma aleatoria, utilizando el 25% de los datos para validación y el 75% restante para entrenamiento.

El código también imprime el rango máximo y mínimo de los datos de imagen, así como el número de dígitos únicos en el conjunto de entrenamiento y el número de muestras y variables en ambos conjuntos.

```
# Cargamos nuestros datos y los separamos en entrenamiento y validación
data, labels = load_digits(return_X_y=True)

# El 25% de los datos se asignará aleatoriamente a validación
data_train, data_val, target_train, target_val = train_test_split(
    data, 
    labels, 
    test_size=0.25
)
print(f"Imágenes en rango {np.max(data)}, {np.min(data)}")

# Entrenamiento
(n_samples, n_features), n_digits = data_train.shape, np.unique(target_train).size
print(f"# Dígitos: {n_digits}; # Muestras de entrenamiento: {n_samples}; # Variables {n_features}")

# Validación
(n_samples, n_features), n_digits = data_val.shape, np.unique(target_val).size
print(f"# Dígitos: {n_digits}; # Muestras de validación: {n_samples}; # Variables {n_features}")

--> Resultados de las impresiones

Imágenes en rango 16.0, 0.0
# Dígitos: 10; # Muestras de entrenamiento: 1347; # Variables 64
# Dígitos: 10; # Muestras de validación: 450; # Variables 64
```

En este proyecto las imágenes se entregan como un vector de 64 variables, donde cada elemento corresponde al valor de un pixel. Para visualizar los datos en forma de imagen, es necesario transformarlos a la forma adecuada. En las siguiente celda puedes ver algunas imágenes de ejemplo, así como la forma en que podemos transformar el vector de variables a una matriz de 8x8.

```
plt.gray()

# Visualizar algunas imágenes
n_cols = 3
idx = np.random.randint(len(data_train), size=n_cols)
fig, axes = plt.subplots(1, n_cols, figsize=(6,3))
axes = axes.flatten()
for ax, i in zip(axes, idx):
    side = np.sqrt(len(data_train[i])).astype('int')
    # La imagen está dada como un solo vector de longitud 64
    # Cambiamos la forma para tenerla en forma de imagen de 8x8 pixeles
    img = data[i].reshape((side, side))
    ax.matshow(img)
    ax.axis('off')
    ax.set_title(f"Etiqueta: {labels[i]}")
fig.suptitle("Ejemplos de muestras de entrenamiento")
plt.tight_layout()
plt.show()
```

El codigo anterior muestra algunos ejemplos de imágenes del conjunto de entrenamiento utilizando la biblioteca de visualización de Matplotlib. Primero, establece la paleta de colores en escala de grises utilizando la función gray() de Matplotlib. Luego, se eligen al azar tres índices de las muestras de entrenamiento utilizando la función np.random.randint, y se crean tres subplots utilizando la función subplots de Matplotlib. Cada imagen se cambia de forma utilizando la función reshape para que tenga una dimensión de 8x8 píxeles, y se muestra utilizando la función matshow.

## Ejemplo de la salida del codigo:

![image](https://user-images.githubusercontent.com/56804608/229397489-df1853cd-46f9-453e-b9bc-ab1f7430dc15.png)

## Visualización en baja dimensionalidad

En el siguiente codigo se visualizaran como se ven los datos reduciendo la dimensionalidad de 30 variables a 2. En este caso nosotros usamos TSNE, pues era el que mejor información nos daba a diferencia del PCA.

```

# Reduciendo la dimensionalidad de los datos de validación data_val a 2 dimensiones usando TSNE

reduced_data_val = TSNE(n_components=2, perplexity=30).fit_transform(data_val)

labels = np.unique(target_train)
fig, ax_pca = plt.subplots(1, 1, figsize=(4,4))
fig.suptitle("Puntos reducidos a dos dimensiones")
for c in labels:
    indices = np.where(target_val == c)
    plot_data = reduced_data_val[indices]
    ax_pca.scatter(plot_data[:, 0], plot_data[:, 1], label=f"Grupo {c}")
plt.show()
```

## Visualización del resultado de este fragmento de codigo

![image](https://user-images.githubusercontent.com/56804608/229398589-5a9edeb7-3a3e-49c6-abb8-95dda647852b.png)

#### Para la imagen anterior, explica detalladamente que información nos da sobre el dataset.

Este fragmento de código utiliza la técnica de reducción de dimensionalidad t-distributed stochastic neighbor embedding (TSNE) del módulo sklearn.manifold para reducir los datos de validación a dos dimensiones y visualizarlos en un gráfico de dispersión. Primero, se llama a la función TSNE para reducir los datos de validación a dos dimensiones, utilizando una perplexidad de 30. Luego, se crea una figura utilizando la función subplots de Matplotlib con un solo subplot. El código utiliza un bucle for para iterar sobre las etiquetas únicas en el conjunto de entrenamiento y crear una subselección de datos correspondiente a cada etiqueta. Luego, se muestra cada subselección de datos en el gráfico de dispersión utilizando la función scatter de Matplotlib. Cada subselección de datos se representa con un color diferente, y cada color se etiqueta con la etiqueta correspondiente en la leyenda del gráfico.

# 2. Funciones de utilidad
```
import scipy as sp
def dbscan_predict(dbscan_model, X_new, metric=sp.spatial.distance.cosine):
    # Result is noise by default
    y_new = np.ones(shape=len(X_new), dtype=int)*-1 
    # Iterate all input samples for a label
    for j, x_new in enumerate(X_new):
        # Find a core sample closer than EPS
        for i, x_core in enumerate(dbscan_model.components_): 
            if metric(x_new, x_core) < dbscan_model.eps:
                # Assign label of x_core to x_new
                y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break
    return y_new
```
En primer lugar, se inicializa un array y_new de forma (nueva longitud de X_new,) y se llena con valores de -1 para indicar que todas las muestras son ruido por defecto. Luego, se itera sobre cada muestra en X_new. Para cada muestra, se itera sobre cada muestra central en dbscan_model.components_, que son las muestras centrales en cada cluster. Si la distancia entre la muestra actual x_new y la muestra central x_core es menor que el umbral dbscan_model.eps, entonces la etiqueta del cluster de la muestra central se asigna a la muestra actual. La función devuelve un array y_new que contiene las etiquetas de cluster asignadas para cada muestra en X_new. Si una muestra no fue asignada a ningún cluster, su etiqueta será -1 (ruido).

# 3. Modelo de agrupamiento

Dados los datos `data_train` con las etiquetas `target_train` definiremos y entrenaremos un algoritmo que identifique los dígitos. 

```
# Instanciamos la normalización de datos.
# Comó validación debe estar completamente disjunto de entrenamiento
# Seleccionamos los valores de normalización usando los datos de entrenamiento
# y aplicamos la misma normalización a ambos
scaler = StandardScaler()
scaler.fit(data_train)
from sklearn.cluster import DBSCAN

def mi_modelo(X, label):
    '''
        args:
            - X (nd.array): Arreglo de dimensionalidad (N, D) donde D=64 conteniendo las imágenes en forma de vector
            - label (nd.array, tipo int): Arreglo de dimensionalidad (N,) conteniendo las etiquetas de clase/grupo para cada imagen
        returns:
            - model (object): Instancia de clase del modelo entrenado en los datos X normalizados
    '''
    # Normalizamos los datos de entrenamiento
    data = scaler.transform(X)

   # Entrenas el modelo y regresamos el modelo entrenado en los datos de entrenamiento.
   # Entrenamos nuestro modelo con los DATOS NORMALIZADOS
    model = LogisticRegression().fit(X, label)

    return model

def mi_inferencia(modelo, X_val):
    '''
        args:
            - modelo(object): Instancia de la clase del modelo que estés utilizando
            - X_val(np.ndarray): Arreglo de dimensionalidad (N, D) donde D=64 conteniendo las imágenes en forma de vector
        returns:
            - preds(np.ndarray, tipo int): Arreglo de dimensionalidad (N,) conteniendo las predicciones de clase/grupo para cada imagen
    '''
    # Normalizamos los datos de validación
    # El mismos preprocesamiento de datos se aplica a
    # tanto inferencia como entrenamiento
    data = scaler.transform(X_val)

    # Utilizando el modelo para predecir valores para los datos de validación
    # Regresa las predicciones de tu modelo para X_val.
    # Aplica inferencia sobre los DATOS NORMALIZADOS
    preds = modelo.predict(data)

    return preds

# Utilizamos solo los datos de entrenamiento (alta dimensionalidad) para entrenar
modelo = mi_modelo(data_train, target_train)

# Utilizamos los datos de validacion (alta dimensionalidad) para hacer inferencia
# con el modelo entrenado
pred = mi_inferencia(modelo, data_val)
```

En el fragmento de código anterior se está entrenando un modelo de regresión logística, utilizando los datos de entrenamiento normalizados y las etiquetas correspondientes a cada imagen. Luego, se está haciendo inferencia con los datos de validación normalizados utilizando el modelo entrenado y se están guardando las predicciones en la variable pred.

Es importante destacar que se está utilizando la misma normalización de datos para el preprocesamiento tanto en el entrenamiento como en la inferencia.

# 4. Evaluación y análisis de las predicciones

En esta sección incluimos funciones que te permiten visualizar la predicción de tu modelo para el set de validación. Dado que nuestros datos son de alta dimensionalidad (64) necesitamos reducirlos para poder analizar las predicciones. Recuerda que en esta sección solo funcionará si has definido tu modelo correctamente en el método anterior `mi_modelo`. Recuerda que en esta sección nos interesa mostar el rendimiento del modelo en los datos de *validación*.

## 4.1 (Inferencia) Predicciones en baja dimensionalidad

```
# Llamamos a inferencia de su modelo
# preds deberá ser de la forma N,1
# indicando el número al que corresponde cada imagen
preds = mi_inferencia(modelo, data_val)

# Buscamos la cantidad de grupos/clases que hay en los datos de validación
group_pred = np.unique(preds)
n_groups = len(group_pred)
print(f"Datos {data_val.shape}, predicciones {preds.shape}, clases/grupos {n_groups}")

fig, ax_reduced = plt.subplots(1, 1, figsize=(4,4))
fig.suptitle("Puntos clasificados reducidos a dos dimensiones")

# Grafícando los datos DE VALIDACIÓN reducidos (reduced_data_val) con un color distinto para cada grupo/clase que nuestro modelo ha predecido (preds)

for group_id in group_pred:
    idx = np.where(target_val == group_id)
    plot_data = reduced_data_val[idx]
    ax_reduced.scatter(plot_data[:, 0], plot_data[:, 1], label=f"Grupo {group_id}")

plt.show()
```
Este código toma el modelo entrenado y lo utiliza para realizar inferencias en los datos de validación. Luego, utiliza las predicciones de clase/grupo resultantes para determinar la cantidad de grupos/clases en los datos de validación y genera un gráfico de dispersión en dos dimensiones de los datos de validación reducidos utilizando TSNE. Cada punto en el gráfico representa una imagen en el conjunto de datos de validación y su color indica a qué grupo/clase se le asignó mediante el modelo entrenado.

# Resultado del codigo anterior

![image](https://user-images.githubusercontent.com/56804608/229400326-3657c927-b4fb-4b10-94d0-2f0abfce36dd.png)

## 4.2 (Inferencia) Visualizar imagenes en cada grupo/clase

El siguiente código llama al método de inferencia anteriormente definido. Se deber mostrar al menos 1 imagen por cada grupo de predicción de tu modelo.

```
# Llamamos a inferencia de nuestro modelo
# Este método regresará las clases/grupos que haya encontrado para cada punto de validación
preds = mi_inferencia(modelo, data_val)

# Clases/grupos distintos de predicciones
group_pred = np.unique(preds)
n_groups = len(group_pred)

# Graficar
n_cols = 5
fig, axes = plt.subplots(n_groups//n_cols, n_cols, figsize=(10,6))  # Puedes cambiar o eliminar esta linea si lo consideras necesario.
axes = axes.flatten() # Convertir los axes a lista para poder iterarlos
# Muestrando una imagen por cada grupo/clase de preddicion

plt.gray()
for ax, i in zip(axes, group_pred):
    side = np.sqrt(len(data_train[i])).astype('int')
    img = data_val[np.where(preds == i)[0][0]].reshape((side, side))
    ax.matshow(img)
    ax.axis('off')
    ax.set_title(f"Numero: {group_pred[i]}")

fig.suptitle("Muestras representativas por grupo")
plt.tight_layout()
plt.show()
```

Este código muestra una imagen representativa por cada grupo/clase de predicción encontrado en los datos de validación.

Primero se llama a la función mi_inferencia para obtener las predicciones de clase/grupo para cada punto de validación. Luego se busca la cantidad de grupos/clases distintos encontrados en las predicciones.

Después se crea una figura con subplots para mostrar las imágenes representativas por grupo. Para cada grupo/clase de predicción, se obtiene la primera imagen correspondiente a ese grupo y se muestra en un subplot. La imagen se obtiene a partir de los datos de validación (data_val) y las predicciones (preds), encontrando el primer punto de validación que pertenece a ese grupo.

Finalmente, se muestra la figura con todas las imágenes representativas por grupo.

![image](https://user-images.githubusercontent.com/56804608/229400790-a992513e-d610-49ce-bf97-dc4261038270.png)

## 4.3 (Inferencia) Calculando el accuracy

En el siguiente codigo se calcula una métrica (un valor numérico) para el conjunto de validación donde indicaremos cuantas muestras en promedio clasifica nuestro modelo correctamente.

```
datos = data_val
etiquetas = target_val
predicciones = mi_inferencia(modelo, datos)

# Calculando el "accuracy" promedio
accuracy = 0
error = np.zeros(len(etiquetas))
count = 0
for i, j in zip(etiquetas, predicciones):
    if (i == j):
        accuracy = accuracy + 1
    else:
        error[count] = 1
    count = count + 1
accuracy = accuracy / len(etiquetas) * 100
print(f"El accuracy de nuestro modelo es de {accuracy}%")

#Graficando Elementos con Predicciones Erroneas
n_elements = 3
idx = np.random.choice(np.where(error==1)[0],n_elements)
plt.gray()
fig, axes = plt.subplots(1, n_elements, figsize=(8,4))
axes = axes.flatten()
for ax, i in zip(axes, idx):
    side = np.sqrt(len(data_val[i])).astype('int')
    img = data[i].reshape((side, side))
    ax.matshow(img)
    ax.axis('off')
    ax.set_title(f"Etiqueta: {etiquetas[i]}, Preddicion: {predicciones[i]}")
fig.suptitle("Predicciones Erroneas")
plt.tight_layout()
plt.show()
```
Este código evalúa el rendimiento del modelo de clasificación entrenado en un conjunto de datos de validación y muestra algunas de las imágenes de validación que fueron clasificadas incorrectamente.

Primero, se hace una predicción utilizando el conjunto de datos de validación y se compara con las etiquetas reales utilizando un bucle for. Se cuenta la cantidad de predicciones correctas y se calcula el porcentaje de precisión del modelo.

Luego, se crea un arreglo de ceros de la misma longitud que las etiquetas reales y se identifican los índices de las imágenes que fueron clasificadas incorrectamente. Se seleccionan tres imágenes al azar de ese subconjunto y se muestran junto con su etiqueta real y la predicción del modelo.

# Resultado de la presición de nuestro modelo

![image](https://user-images.githubusercontent.com/56804608/229401194-056204b6-f815-42ae-a2ba-02368e9d6645.png)

# Resultados erroneos

![image](https://user-images.githubusercontent.com/56804608/229401238-c2f75d7c-4f9b-4a24-8830-c5a54a553776.png)


# eliasib18.github.io
