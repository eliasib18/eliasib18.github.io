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

# eliasib18.github.io
