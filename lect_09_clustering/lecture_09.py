import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# =========================
# Configuración general
# =========================
# Semilla aleatoria para garantizar reproducibilidad en KMeans y PCA
random_state = 42
# Usamos fuente serif para gráficas con estilo más formal/académico
plt.rc('font', family='serif', size=12)

# =========================
# 1. Cargar dataset
# =========================
# Se lee el CSV generado sintéticamente para el curso FIRE-UdeA.
# Asegúrate de que el archivo esté en el mismo directorio de trabajo.
ruta = "dataset_sintetico_FIRE_UdeA.csv"
df = pd.read_csv(ruta)

print("Primeras filas del dataset:")
print(df.head())
print("\nDimensiones:", df.shape)
print("\nColumnas:", df.columns.tolist())

# =========================
# 2. Separar variables
# =========================
# En aprendizaje no supervisado NO usamos la etiqueta para entrenar.
# Si existe la columna "label", la guardamos solo para comparar resultados al final
# y la excluimos del conjunto de entrada X.
if "label" in df.columns:
    y_real = df["label"]   # etiqueta real: solo para evaluación posterior
    X = df.drop(columns=["label"])
else:
    y_real = None
    X = df.copy()

print("\nVariables usadas para clustering:")
print(X.columns.tolist())

# =========================
# 3. Preprocesamiento
# =========================
# Como todas las columnas son numéricas en este dataset, aplicamos únicamente
# StandardScaler, que transforma cada variable para tener media 0 y desviación 1.
# Esto es crucial en clustering: sin escalar, variables con rangos grandes dominan
# la distancia y sesgan los resultados.
numeric_features = X.columns.tolist()

numeric_transformer = Pipeline(
    steps=[("scaler", StandardScaler())]
)

# ColumnTransformer permite aplicar distintas transformaciones a distintos subconjuntos
# de columnas. Aquí solo tenemos numéricas, pero la estructura queda lista para extender.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
    ],
    remainder="drop"  # cualquier columna no listada se descarta
)

# =========================
# 4. Visualización 2D con PCA
# =========================
# Escalamos los datos antes de aplicar PCA. PCA es sensible a la escala,
# por eso siempre debe aplicarse sobre datos ya normalizados.
X_scaled = preprocessor.fit_transform(X)

# PCA reduce la dimensionalidad a 2 componentes principales para poder
# visualizar los datos en un plano 2D. Los ejes no tienen unidades originales;
# representan las direcciones de mayor varianza en los datos.
pca = PCA(n_components=2, random_state=random_state)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots()
ax.scatter(X_pca[:, 0], X_pca[:, 1])
ax.set_title("Datos proyectados en 2D con PCA")
ax.set_xlabel("Componente principal 1")
ax.set_ylabel("Componente principal 2")
fig.set_size_inches(8, 5)
plt.show()

# =========================
# 5. Método del codo + Silhouette con KMeans
# =========================
# Probamos distintos valores de k (de 2 a 10) para encontrar el número óptimo
# de clústeres. Usamos dos criterios complementarios:
#   - Inercia (método del codo): suma de distancias cuadradas al centroide.
#     Queremos encontrar el "codo" donde la inercia deja de bajar significativamente.
#   - Silhouette Score: mide qué tan bien separados están los clústeres.
#     Varía entre -1 y 1; valores más cercanos a 1 son mejores.
inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    # Construimos un Pipeline que incluye preprocesamiento + KMeans.
    # n_init=10 significa que KMeans se ejecuta 10 veces con centros iniciales
    # distintos y conserva la mejor solución (menor inercia).
    clu_kmeans = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clustering", KMeans(n_clusters=k, random_state=random_state, n_init=10))
    ])
    
    clu_kmeans.fit(X)
    
    labels_k = clu_kmeans["clustering"].labels_
    inertia_k = clu_kmeans["clustering"].inertia_
    # El silhouette se calcula sobre los datos ya escalados para respetar las distancias
    sil_k = silhouette_score(preprocessor.transform(X), labels_k)
    
    inertias.append(inertia_k)
    silhouette_scores.append(sil_k)

# Gráfica del método del codo: buscar el punto donde la curva "dobla"
fig, ax = plt.subplots()
ax.plot(list(k_range), inertias, marker='o')
ax.set_title("Método del codo - KMeans")
ax.set_xlabel("Número de clústeres (k)")
ax.set_ylabel("Inercia")
fig.set_size_inches(8, 5)
plt.show()

# Gráfica del silhouette: buscar el k con el valor más alto
fig, ax = plt.subplots()
ax.plot(list(k_range), silhouette_scores, marker='o')
ax.set_title("Silhouette Score - KMeans")
ax.set_xlabel("Número de clústeres (k)")
ax.set_ylabel("Silhouette")
fig.set_size_inches(8, 5)
plt.show()

# =========================
# 6. Entrenar KMeans con el k elegido
# =========================
# Ajusta este valor según lo que indiquen el codo y el silhouette.
# Aquí se usa k=2 como punto de partida; cámbialo según tus observaciones.
k_optimo = 2

clu_kmeans = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clustering", KMeans(n_clusters=k_optimo, random_state=random_state, n_init=10))
])

clu_kmeans.fit(X)
labels_kmeans = clu_kmeans["clustering"].labels_

print(f"\nKMeans con k = {k_optimo}")
print("Inercia:", clu_kmeans["clustering"].inertia_)
print("Silhouette:", silhouette_score(preprocessor.transform(X), labels_kmeans))

# Visualizamos los clústeres asignados por KMeans en el espacio PCA 2D.
# Cada color representa un clúster distinto.
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans)
ax.set_title(f"KMeans con k = {k_optimo}")
ax.set_xlabel("Componente principal 1")
ax.set_ylabel("Componente principal 2")
fig.set_size_inches(8, 5)
plt.show()

# =========================
# 7. Entrenar DBSCAN
# =========================
# DBSCAN (Density-Based Spatial Clustering of Applications with Noise) no requiere
# especificar k de antemano. En cambio, agrupa puntos que están suficientemente
# cerca entre sí (eps) y que forman vecindades densas (min_samples).
# Los puntos que no pertenecen a ningún clúster se etiquetan como ruido (-1).
#
# Parámetros clave:
#   eps        : radio de vecindad. Puntos dentro de este radio se consideran vecinos.
#   min_samples: mínimo de puntos para que un punto sea considerado "núcleo".
#
# Sugerencia: ajusta eps usando una gráfica de k-distancias si los resultados no son buenos.
clu_dbscan = DBSCAN(eps=0.8, min_samples=10)
labels_dbscan = clu_dbscan.fit_predict(X_scaled)

print("\nDBSCAN")
print("Etiquetas encontradas:", np.unique(labels_dbscan))
print("Conteo por etiqueta:", np.unique(labels_dbscan, return_counts=True))

# El silhouette de DBSCAN se calcula excluyendo los puntos de ruido (-1),
# ya que no pertenecen a ningún clúster y distorsionarían la métrica.
labels_unicos = set(labels_dbscan)
if len(labels_unicos - {-1}) > 1:
    mask = labels_dbscan != -1
    sil_db = silhouette_score(X_scaled[mask], labels_dbscan[mask])
    print("Silhouette DBSCAN (sin ruido):", sil_db)
else:
    print("DBSCAN no encontró suficientes clústeres para calcular silhouette.")

fig, ax = plt.subplots()
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_dbscan)
ax.set_title("DBSCAN")
ax.set_xlabel("Componente principal 1")
ax.set_ylabel("Componente principal 2")
fig.set_size_inches(8, 5)
plt.show()

# =========================
# 8. Comparación con etiqueta real
# =========================
# Esta sección es SOLO para evaluación visual: comparamos los clústeres obtenidos
# contra las etiquetas reales del dataset sintético.
# En un caso real de clustering no supervisado, estas etiquetas no existen.
if y_real is not None:
    fig, ax = plt.subplots()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_real)
    ax.set_title("Etiqueta real del dataset (solo comparación)")
    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    fig.set_size_inches(8, 5)
    plt.show()
