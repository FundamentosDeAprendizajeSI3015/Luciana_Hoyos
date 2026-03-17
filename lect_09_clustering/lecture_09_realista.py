import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer

# =========================
# Configuración general
# =========================
# Semilla aleatoria para reproducibilidad en KMeans, PCA y cualquier proceso estocástico
random_state = 42
# Estilo de fuente para gráficas con apariencia académica/formal
plt.rc('font', family='serif', size=12)

# =========================
# 1. Cargar dataset
# =========================
# Este dataset "realista" incluye variables mixtas (numéricas y categóricas)
# y puede contener valores faltantes, a diferencia de la versión simple.
ruta = "dataset_sintetico_FIRE_UdeA_realista.csv"
df = pd.read_csv(ruta)

print("Primeras filas del dataset:")
print(df.head())
print("\nDimensiones:", df.shape)
print("\nColumnas:", df.columns.tolist())
print("\nTipos de datos:")
print(df.dtypes)

# Es importante revisar los valores faltantes antes de cualquier transformación.
# Un alto porcentaje de NaN en una columna puede requerir eliminarla o imputarla con cuidado.
print("\nValores faltantes por columna:")
print(df.isna().sum())

# =========================
# 2. Separar variables
# =========================
# En aprendizaje no supervisado, la etiqueta NO participa en el entrenamiento.
# Si existe "label", la conservamos aparte para comparación visual al final.
if "label" in df.columns:
    y_real = df["label"]
    X = df.drop(columns=["label"])
else:
    y_real = None
    X = df.copy()

print("\nVariables usadas para clustering:")
print(X.columns.tolist())

# =========================
# 3. Identificar columnas numéricas y categóricas
# =========================
# sklearn distingue automáticamente los tipos usando dtypes de pandas.
# Las variables de tipo "object" o "category" se tratan como categóricas.
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

print("\nVariables numéricas:", numeric_features)
print("Variables categóricas:", categorical_features)

# =========================
# 4. Preprocesamiento
# =========================
# Para variables numéricas:
#   1. SimpleImputer rellena los NaN con la media de cada columna.
#   2. StandardScaler normaliza a media 0 y desviación 1.
#      Esto es esencial para clustering basado en distancias (KMeans, DBSCAN).
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Para variables categóricas:
#   1. SimpleImputer rellena los NaN con el valor más frecuente (moda).
#   2. OneHotEncoder convierte cada categoría en columnas binarias (0/1).
#      handle_unknown="ignore" evita errores si aparecen categorías nuevas.
#      sparse_output=False devuelve un array denso, más fácil de manejar.
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# ColumnTransformer aplica cada pipeline al subconjunto de columnas correspondiente
# y concatena horizontalmente los resultados en una sola matriz.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="drop"  # columnas no listadas se descartan
)

# =========================
# 5. Transformar datos
# =========================
# Ajustamos y transformamos en un solo paso. La matriz resultante es densa,
# combinando las columnas numéricas escaladas y las columnas one-hot de las categóricas.
X_processed = preprocessor.fit_transform(X)

print("\nForma de la matriz transformada:", X_processed.shape)
# Verificación de calidad: no debería haber NaN después de imputar
print("¿Hay NaN después del preprocesamiento?:", np.isnan(X_processed).sum())

# =========================
# 6. Visualización 2D con PCA
# =========================
# Reducimos la dimensionalidad a 2 para poder visualizar los datos.
# PCA encuentra las dos direcciones ortogonales de mayor varianza.
# Nota: siempre aplicar PCA DESPUÉS de escalar los datos.
pca = PCA(n_components=2, random_state=random_state)
X_pca = pca.fit_transform(X_processed)

fig, ax = plt.subplots()
ax.scatter(X_pca[:, 0], X_pca[:, 1])
ax.set_title("Datos proyectados en 2D con PCA")
ax.set_xlabel("Componente principal 1")
ax.set_ylabel("Componente principal 2")
fig.set_size_inches(8, 5)
plt.show()

# =========================
# 7. Método del codo + Silhouette con KMeans
# =========================
# Evaluamos k de 2 a 10 usando dos métricas complementarias:
#   - Inercia: suma de distancias cuadradas de cada punto a su centroide.
#     Una caída brusca seguida de una meseta sugiere el k óptimo ("codo").
#   - Silhouette Score: combina cohesión intra-clúster y separación inter-clúster.
#     Rango [-1, 1]; el k con el valor más alto es el más adecuado.
inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    # n_init=10: KMeans se reinicia 10 veces con distintas semillas y conserva
    # la solución con menor inercia, reduciendo el riesgo de mínimos locales.
    modelo_kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels_k = modelo_kmeans.fit_predict(X_processed)
    inertias.append(modelo_kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_processed, labels_k))

# Gráfica del codo: el k óptimo suele estar en el punto de inflexión de la curva
fig, ax = plt.subplots()
ax.plot(list(k_range), inertias, marker='o')
ax.set_title("Método del codo - KMeans")
ax.set_xlabel("Número de clústeres (k)")
ax.set_ylabel("Inercia")
fig.set_size_inches(8, 5)
plt.show()

# Gráfica de silhouette: pico más alto = mejor separación entre clústeres
fig, ax = plt.subplots()
ax.plot(list(k_range), silhouette_scores, marker='o')
ax.set_title("Silhouette Score - KMeans")
ax.set_xlabel("Número de clústeres (k)")
ax.set_ylabel("Silhouette")
fig.set_size_inches(8, 5)
plt.show()

# =========================
# 8. Elegir el mejor k automáticamente
# =========================
# Seleccionamos el k que maximiza el silhouette score.
# En la práctica, conviene también inspeccionar la gráfica del codo manualmente,
# ya que el silhouette puede favorecer k pequeños aunque el codo sugiera otro.
k_optimo = list(k_range)[np.argmax(silhouette_scores)]
print(f"\nMejor k según silhouette: {k_optimo}")

# =========================
# 9. Entrenar KMeans final
# =========================
# Entrenamos el modelo definitivo con el k óptimo encontrado.
modelo_kmeans_final = KMeans(
    n_clusters=k_optimo,
    random_state=random_state,
    n_init=10
)

labels_kmeans = modelo_kmeans_final.fit_predict(X_processed)

print(f"\nKMeans con k = {k_optimo}")
print("Inercia:", modelo_kmeans_final.inertia_)
print("Silhouette:", silhouette_score(X_processed, labels_kmeans))

# Visualizamos los clústeres finales en el espacio 2D de PCA.
# Cada color corresponde a un clúster; la forma puede diferir de la realidad
# porque PCA es una proyección lineal que no captura toda la varianza.
fig, ax = plt.subplots()
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans)
ax.set_title(f"KMeans con k = {k_optimo}")
ax.set_xlabel("Componente principal 1")
ax.set_ylabel("Componente principal 2")
fig.set_size_inches(8, 5)
plt.show()

# =========================
# 10. DBSCAN
# =========================
# DBSCAN detecta clústeres de forma arbitraria y maneja ruido de forma natural.
# No requiere definir k, pero sí dos hiperparámetros:
#   eps        : distancia máxima entre dos puntos para ser vecinos.
#   min_samples: número mínimo de puntos para formar una región densa (núcleo).
#
# Los puntos que no alcanzan la densidad mínima se etiquetan como ruido (-1).
# Recomendación: grafica la curva de k-distancias (k=min_samples) para elegir eps.
modelo_dbscan = DBSCAN(eps=1.2, min_samples=5)
labels_dbscan = modelo_dbscan.fit_predict(X_processed)

print("\nDBSCAN")
print("Etiquetas encontradas:", np.unique(labels_dbscan))
print("Conteo por etiqueta:", np.unique(labels_dbscan, return_counts=True))

# El silhouette excluye los puntos de ruido (-1) porque no forman parte de
# ningún clúster y distorsionarían la medida de cohesión/separación.
labels_unicos = set(labels_dbscan)
if len(labels_unicos - {-1}) > 1:
    mask = labels_dbscan != -1
    sil_db = silhouette_score(X_processed[mask], labels_dbscan[mask])
    print("Silhouette DBSCAN (sin ruido):", sil_db)
else:
    print("DBSCAN no encontró suficientes clústeres válidos para calcular silhouette.")

fig, ax = plt.subplots()
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_dbscan)
ax.set_title("DBSCAN")
ax.set_xlabel("Componente principal 1")
ax.set_ylabel("Componente principal 2")
fig.set_size_inches(8, 5)
plt.show()

# =========================
# 11. Comparación con la etiqueta real
# =========================
# Comparación ÚNICAMENTE visual y educativa: cotejamos los clústeres obtenidos
# con la etiqueta real del dataset sintético para medir qué tan bien se recuperó
# la estructura subyacente. En un problema real no supervisado esto no existe.
if y_real is not None:
    fig, ax = plt.subplots()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_real)
    ax.set_title("Etiqueta real del dataset (solo comparación)")
    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    fig.set_size_inches(8, 5)
    plt.show()

# =========================
# 12. Guardar resultados
# =========================
# Adjuntamos al dataframe original las etiquetas de clúster asignadas por cada modelo.
# Esto permite hacer análisis de perfil de clústeres posteriormente
# (p. ej., estadísticas descriptivas por clúster, exportar a CSV, etc.).
df_resultado = df.copy()
df_resultado["cluster_kmeans"] = labels_kmeans
df_resultado["cluster_dbscan"] = labels_dbscan

print("\nPrimeras filas con clusters asignados:")
print(df_resultado.head())
