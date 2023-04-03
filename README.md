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
```

# eliasib18.github.io
