# ==============================================================
# INFORME TEÓRICO-PRÁCTICO 02 — Machine Learning
# Dataset: dataset_desercion_estudiantes.csv
# Target: Desertó (No = 0, Sí = 1)
# ==============================================================
# PARTE 1 — Análisis NO supervisado
#           KMeans, Fuzzy C-Means, Subtractive Clustering,
#           DBSCAN, Familia de clustering
# PARTE 2 — Re-evaluación de etiquetas (~30% pueden estar mal)
# PARTE 3 — Modelos supervisados con etiquetas re-evaluadas
#           Árbol de Decisión, Regresión Logística, Regresión Lineal
# PARTE 4 — Comparación: dataset original vs re-etiquetado
# ==============================================================

import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, Birch
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score,
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, mean_squared_error, r2_score
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression, LinearRegression

# ==============================================================
# CONFIGURACIÓN GLOBAL
# ==============================================================

SEED = 42
OUTDIR = Path("./output_desercion")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Paleta oscura personalizada
PALETTE = {
    "bg":      "#1a1a2e",
    "panel":   "#16213e",
    "accent1": "#e94560",
    "accent2": "#0f3460",
    "accent3": "#533483",
    "accent4": "#f5a623",
    "text":    "#e0e0e0",
    "grid":    "#2a2a4a",
}

CMAP_CLUSTERS = "plasma"
CMAP_HEAT     = "magma"

def estilo_oscuro(fig, axes_list=None):
    fig.patch.set_facecolor(PALETTE["bg"])
    if axes_list:
        for ax in axes_list:
            ax.set_facecolor(PALETTE["panel"])
            ax.tick_params(colors=PALETTE["text"])
            ax.xaxis.label.set_color(PALETTE["text"])
            ax.yaxis.label.set_color(PALETTE["text"])
            ax.title.set_color(PALETTE["text"])
            for spine in ax.spines.values():
                spine.set_edgecolor(PALETTE["grid"])
            ax.grid(True, color=PALETTE["grid"], linestyle="--", alpha=0.5)

def guardar(fig, nombre):
    ruta = OUTDIR / nombre
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [✓] {ruta}")

# ==============================================================
# CARGA Y PREPROCESAMIENTO
# ==============================================================

print("\n" + "━"*60)
print("  CARGANDO DATASET — Deserción Estudiantil")
print("━"*60)

data = pd.read_csv("dataset_desercion_estudiantes.csv")
print(data.head())
print(f"\nDimensiones: {data.shape}")

# Codificar target y variable categórica
le = LabelEncoder()
data["Deserto_num"] = le.fit_transform(data["Desertó"])   # No=0, Sí=1
data["Becado_num"]  = le.fit_transform(data["Becado"])    # No=0, Sí=1

print("\nDistribución del target original:")
print(data["Desertó"].value_counts())

X_raw = data[["Promedio", "Materias_Perdidas", "Becado_num"]].copy()
y_original = data["Deserto_num"].values

num_cols = ["Promedio", "Materias_Perdidas"]
cat_cols = []   # ya codificamos Becado manualmente como numérico

# Pipeline de preprocesamiento
preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])

X_proc = preprocessor.fit_transform(X_raw)
print(f"\nMatriz preprocesada: {X_proc.shape}")

# PCA 2D para visualizaciones
pca2 = PCA(n_components=2, random_state=SEED)
X_pca = pca2.fit_transform(X_proc)

print(f"Varianza explicada PCA: {pca2.explained_variance_ratio_.cumsum()[-1]:.1%}")

# ==============================================================
# VISTA GENERAL DEL DATASET (figura extra introductoria)
# ==============================================================

fig = plt.figure(figsize=(16, 5), facecolor=PALETTE["bg"])
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

ax0 = fig.add_subplot(gs[0])
counts = data["Desertó"].value_counts()
wedge_colors = [PALETTE["accent1"], PALETTE["accent2"]]
ax0.pie(counts, labels=counts.index, colors=wedge_colors,
        autopct="%1.1f%%", textprops={"color": PALETTE["text"]},
        wedgeprops={"edgecolor": PALETTE["bg"], "linewidth": 2})
ax0.set_title("Distribución: Desertó", color=PALETTE["text"], fontsize=12)
ax0.set_facecolor(PALETTE["panel"])

ax1 = fig.add_subplot(gs[1])
for label, color in zip(["No", "Sí"], [PALETTE["accent2"], PALETTE["accent1"]]):
    subset = data[data["Desertó"] == label]["Promedio"]
    ax1.hist(subset, bins=20, alpha=0.75, color=color, label=label, edgecolor=PALETTE["bg"])
ax1.set_title("Distribución de Promedios", color=PALETTE["text"], fontsize=12)
ax1.set_xlabel("Promedio"); ax1.set_ylabel("Frecuencia")
ax1.legend(title="Desertó", facecolor=PALETTE["panel"], labelcolor=PALETTE["text"])
estilo_oscuro(fig, [ax1])

ax2 = fig.add_subplot(gs[2])
sns.boxplot(data=data, x="Desertó", y="Materias_Perdidas",
            palette={"No": PALETTE["accent2"], "Sí": PALETTE["accent1"]},
            ax=ax2, width=0.5, linewidth=1.5,
            boxprops=dict(edgecolor=PALETTE["text"]),
            whiskerprops=dict(color=PALETTE["text"]),
            capprops=dict(color=PALETTE["text"]),
            medianprops=dict(color=PALETTE["accent4"], linewidth=2))
ax2.set_title("Materias Perdidas vs Deserción", color=PALETTE["text"], fontsize=12)
estilo_oscuro(fig, [ax2])

fig.suptitle("Análisis Exploratorio — Dataset Deserción Estudiantil",
             color=PALETTE["text"], fontsize=14, y=1.02)
guardar(fig, "00_eda_general.png")

# ==============================================================
# PARTE 1 — ANÁLISIS NO SUPERVISADO
# ==============================================================

print("\n" + "━"*60)
print("  PARTE 1 — ANÁLISIS NO SUPERVISADO")
print("━"*60)

# ----------------------------------------------------------
# 1A. KMeans — Codo + Silhouette (gráfica estilo radar/línea)
# ----------------------------------------------------------
print("\n[1A] KMeans — codo + silhouette...")

inertias, sil_scores = [], []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    lbl = km.fit_predict(X_proc)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_proc, lbl))

fig = plt.figure(figsize=(13, 5), facecolor=PALETTE["bg"])
ax_l = fig.add_subplot(121)
ax_r = fig.add_subplot(122)

ks = list(k_range)
ax_l.fill_between(ks, inertias, alpha=0.25, color=PALETTE["accent1"])
ax_l.plot(ks, inertias, "o-", color=PALETTE["accent1"], linewidth=2, markersize=7)
ax_l.set_title("KMeans — Método del Codo", fontsize=12)
ax_l.set_xlabel("Número de clústeres k")
ax_l.set_ylabel("Inercia (WCSS)")

ax_r.fill_between(ks, sil_scores, alpha=0.25, color=PALETTE["accent4"])
ax_r.plot(ks, sil_scores, "s-", color=PALETTE["accent4"], linewidth=2, markersize=7)
ax_r.set_title("KMeans — Silhouette Score", fontsize=12)
ax_r.set_xlabel("Número de clústeres k")
ax_r.set_ylabel("Silhouette Score")

estilo_oscuro(fig, [ax_l, ax_r])
fig.tight_layout()
guardar(fig, "01_kmeans_codo_silhouette.png")

k_optimo = ks[np.argmax(sil_scores)]
print(f"  → Mejor k según silhouette: {k_optimo}")

km_final = KMeans(n_clusters=k_optimo, random_state=SEED, n_init=10)
labels_kmeans = km_final.fit_predict(X_proc)
sil_km = silhouette_score(X_proc, labels_kmeans)
print(f"  Silhouette final KMeans: {sil_km:.4f}")

# Scatter PCA con hexágonos de centroide
centros_pca = pca2.transform(km_final.cluster_centers_)
fig, ax = plt.subplots(figsize=(8, 6), facecolor=PALETTE["bg"])
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans,
                cmap=CMAP_CLUSTERS, s=50, alpha=0.8, edgecolors="none")
ax.scatter(centros_pca[:, 0], centros_pca[:, 1], marker="*",
           s=300, c="white", zorder=5, edgecolors=PALETTE["accent4"], linewidths=1.5)
ax.set_title(f"KMeans — k={k_optimo}  |  Sil={sil_km:.3f}", fontsize=12)
ax.set_xlabel("Componente Principal 1")
ax.set_ylabel("Componente Principal 2")
plt.colorbar(sc, ax=ax, label="Cluster").ax.yaxis.label.set_color(PALETTE["text"])
estilo_oscuro(fig, [ax])
guardar(fig, "02_kmeans_clusters_pca.png")

# ----------------------------------------------------------
# 1B. Fuzzy C-Means (implementación manual)
# ----------------------------------------------------------
print("\n[1B] Fuzzy C-Means (manual)...")

def fuzzy_cmeans(X, c, m=2, max_iter=300, tol=1e-6, seed=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    U = rng.random((n, c))
    U = U / U.sum(axis=1, keepdims=True)
    for _ in range(max_iter):
        Um = U ** m
        centers = (Um.T @ X) / Um.sum(axis=0)[:, None]
        dist = np.array([np.linalg.norm(X - centers[j], axis=1) for j in range(c)]).T
        dist = np.maximum(dist, 1e-10)
        U_new = np.zeros_like(U)
        for j in range(c):
            ratio = (dist[:, j:j+1] / dist) ** (2 / (m - 1))
            U_new[:, j] = 1.0 / ratio.sum(axis=1)
        if np.linalg.norm(U_new - U) < tol:
            break
        U = U_new
    fpc = np.trace(U_new.T @ U_new) / n
    return U_new, centers, fpc

U_fcm, centers_fcm, fpc = fuzzy_cmeans(X_proc, c=k_optimo, seed=SEED)
labels_fcm = np.argmax(U_fcm, axis=1)
sil_fcm    = silhouette_score(X_proc, labels_fcm)
print(f"  Silhouette FCM: {sil_fcm:.4f}  |  FPC: {fpc:.4f}")

# Mapa de calor de membresías (primeros 60 puntos)
fig = plt.figure(figsize=(12, 5), facecolor=PALETTE["bg"])
ax_sc = fig.add_subplot(121)
ax_hm = fig.add_subplot(122)

sc2 = ax_sc.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_fcm,
                    cmap=CMAP_CLUSTERS, s=50, alpha=0.8, edgecolors="none")
ax_sc.set_title(f"Fuzzy C-Means — c={k_optimo}  |  Sil={sil_fcm:.3f}", fontsize=11)
ax_sc.set_xlabel("CP1"); ax_sc.set_ylabel("CP2")
plt.colorbar(sc2, ax=ax_sc, label="Cluster").ax.yaxis.label.set_color(PALETTE["text"])

sns.heatmap(U_fcm[:60].T, ax=ax_hm, cmap="YlOrRd",
            xticklabels=False, cbar_kws={"label": "Membresía"},
            linewidths=0.2, linecolor=PALETTE["bg"])
ax_hm.set_title("Membresías (primeros 60 puntos)", fontsize=11)
ax_hm.set_ylabel("Cluster"); ax_hm.set_xlabel("Muestra")
ax_hm.title.set_color(PALETTE["text"])
ax_hm.tick_params(colors=PALETTE["text"])

estilo_oscuro(fig, [ax_sc])
fig.patch.set_facecolor(PALETTE["bg"])
fig.tight_layout()
guardar(fig, "03_fuzzy_cmeans.png")

# ----------------------------------------------------------
# 1C. Subtractive Clustering
# ----------------------------------------------------------
print("\n[1C] Subtractive Clustering...")

def subtractive_clustering(X, ra=0.5, rb_factor=1.5, eps_up=0.5, eps_down=0.15):
    rb = ra * rb_factor
    X_s = (X - X.min(axis=0)) / (np.ptp(X, axis=0) + 1e-10)
    n   = X_s.shape[0]
    alpha = 4 / (ra ** 2)
    beta  = 4 / (rb ** 2)
    pot = np.array([
        np.exp(-alpha * np.sum((X_s - X_s[i]) ** 2, axis=1)).sum()
        for i in range(n)
    ])
    centers_idx = []
    pot_max0 = pot.max()
    while True:
        i_best  = np.argmax(pot)
        p_best  = pot[i_best]
        thr_up  = eps_up  * pot_max0
        thr_dn  = eps_down * pot_max0
        if p_best >= thr_up:
            centers_idx.append(i_best)
        elif p_best <= thr_dn:
            break
        else:
            if centers_idx:
                d_min = min(np.linalg.norm(X_s[i_best] - X_s[c]) for c in centers_idx)
                if (d_min / ra) + (p_best / pot_max0) >= 1:
                    centers_idx.append(i_best)
                else:
                    pot[i_best] = 0
                    continue
            else:
                centers_idx.append(i_best)
        pot -= p_best * np.exp(-beta * np.sum((X_s - X_s[i_best]) ** 2, axis=1))
        pot[i_best] = 0
        if len(centers_idx) > 30:
            break
    centers_idx = list(dict.fromkeys(centers_idx))
    if not centers_idx:
        centers_idx = [0]
    centers = X_s[centers_idx]
    labels_s = np.argmin(
        np.array([np.linalg.norm(X_s - c, axis=1) for c in centers]), axis=0
    )
    return labels_s, centers_idx

labels_sub, idx_sub = subtractive_clustering(X_proc, ra=0.5)
n_sub = len(np.unique(labels_sub))
sil_sub = silhouette_score(X_proc, labels_sub) if n_sub > 1 else None
sil_sub_str = f"{sil_sub:.4f}" if sil_sub is not None else "N/A"
print(f"  Centros detectados: {n_sub}  |  Silhouette: {sil_sub_str}")

# Voronoi-style scatter con tamaño ∝ membresía al propio centro
fig, ax = plt.subplots(figsize=(8, 6), facecolor=PALETTE["bg"])
sc3 = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_sub,
                 cmap="cool", s=45, alpha=0.85, edgecolors="none")
# Marcar los centros reales en espacio PCA
for ci in idx_sub:
    ax.annotate("★", (X_pca[ci, 0], X_pca[ci, 1]),
                color=PALETTE["accent4"], fontsize=14, ha="center", va="center")
ax.set_title(f"Subtractive Clustering — {n_sub} centros", fontsize=12)
ax.set_xlabel("CP1"); ax.set_ylabel("CP2")
plt.colorbar(sc3, ax=ax, label="Cluster").ax.yaxis.label.set_color(PALETTE["text"])
estilo_oscuro(fig, [ax])
guardar(fig, "04_subtractive_clustering.png")

# ----------------------------------------------------------
# 1D. DBSCAN
# ----------------------------------------------------------
print("\n[1D] DBSCAN...")

dbscan = DBSCAN(eps=1.0, min_samples=5)
labels_db = dbscan.fit_predict(X_proc)
n_db    = len(set(labels_db) - {-1})
n_noise = (labels_db == -1).sum()
print(f"  Clústeres: {n_db}  |  Ruido: {n_noise}")

sil_db = None
if n_db > 1:
    mask = labels_db != -1
    sil_db = silhouette_score(X_proc[mask], labels_db[mask])
    print(f"  Silhouette (sin ruido): {sil_db:.4f}")

# Gráfica: clústeres normales + ruido resaltado
fig, ax = plt.subplots(figsize=(8, 6), facecolor=PALETTE["bg"])
mask_core  = labels_db != -1
mask_noise = labels_db == -1
sc4 = ax.scatter(X_pca[mask_core, 0], X_pca[mask_core, 1],
                 c=labels_db[mask_core], cmap=CMAP_CLUSTERS,
                 s=50, alpha=0.9, edgecolors="none", label="Core / Border")
ax.scatter(X_pca[mask_noise, 0], X_pca[mask_noise, 1],
           marker="x", c=PALETTE["accent1"], s=60, linewidths=1.5,
           alpha=0.9, label=f"Ruido ({n_noise})")
ax.set_title(f"DBSCAN — {n_db} clústeres  |  ε=1.0  min_samples=5", fontsize=12)
ax.set_xlabel("CP1"); ax.set_ylabel("CP2")
ax.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"])
plt.colorbar(sc4, ax=ax, label="Cluster").ax.yaxis.label.set_color(PALETTE["text"])
estilo_oscuro(fig, [ax])
guardar(fig, "05_dbscan.png")

# ----------------------------------------------------------
# 1E. Familia de clustering
# ----------------------------------------------------------
print("\n[1E] Familia de clustering...")

cluster_family = {
    "Agg. Ward":     AgglomerativeClustering(n_clusters=k_optimo, linkage="ward"),
    "Agg. Average":  AgglomerativeClustering(n_clusters=k_optimo, linkage="average"),
    "Agg. Complete": AgglomerativeClustering(n_clusters=k_optimo, linkage="complete"),
    "Spectral":      SpectralClustering(n_clusters=k_optimo, random_state=SEED, affinity="rbf"),
    "BIRCH":         Birch(n_clusters=k_optimo),
}

sil_family  = {}
lbl_family  = {}

fig_f = plt.figure(figsize=(20, 10), facecolor=PALETTE["bg"])
gs_f  = gridspec.GridSpec(2, 3, figure=fig_f, hspace=0.45, wspace=0.35)

for idx, (name, model) in enumerate(cluster_family.items()):
    lbl = model.fit_predict(X_proc)
    sil = silhouette_score(X_proc, lbl)
    sil_family[name] = sil
    lbl_family[name] = lbl
    print(f"  {name:20s}: Silhouette={sil:.4f}")

    ax_f = fig_f.add_subplot(gs_f[idx // 3, idx % 3])
    sc_f = ax_f.scatter(X_pca[:, 0], X_pca[:, 1], c=lbl,
                        cmap=CMAP_CLUSTERS, s=30, alpha=0.8, edgecolors="none")
    ax_f.set_title(f"{name}  |  Sil={sil:.3f}", fontsize=10)
    ax_f.set_xlabel("CP1"); ax_f.set_ylabel("CP2")
    estilo_oscuro(fig_f, [ax_f])

# Radar de silhouettes
ax_bar2 = fig_f.add_subplot(gs_f[1, 2])
names_b = list(sil_family.keys())
vals_b  = list(sil_family.values())
colors_b = [PALETTE["accent1"], PALETTE["accent2"], PALETTE["accent3"],
            PALETTE["accent4"], "#5de0e6"]
bars2 = ax_bar2.barh(names_b, vals_b, color=colors_b, edgecolor=PALETTE["bg"], height=0.6)
for bar, v in zip(bars2, vals_b):
    ax_bar2.text(v + 0.002, bar.get_y() + bar.get_height()/2,
                 f"{v:.3f}", va="center", color=PALETTE["text"], fontsize=9)
ax_bar2.set_xlabel("Silhouette Score")
ax_bar2.set_title("Comparación — Familia", fontsize=10)
ax_bar2.set_xlim(0, max(vals_b) * 1.2)
estilo_oscuro(fig_f, [ax_bar2])
guardar(fig_f, "06_familia_clustering.png")

# ----------------------------------------------------------
# 1F. Resumen comparativo clustering (gráfica de burbujas)
# ----------------------------------------------------------
print("\n[1F] Resumen comparativo clustering...")

resumen_sil = {
    "KMeans":        sil_km,
    "Fuzzy C-Means": sil_fcm,
    "Subtractive":   sil_sub if sil_sub else np.nan,
    "DBSCAN":        sil_db  if sil_db  else np.nan,
}
resumen_sil.update(sil_family)

df_sil = pd.DataFrame.from_dict(resumen_sil, orient="index", columns=["Silhouette"])
df_sil = df_sil.sort_values("Silhouette", ascending=False)
print(df_sil.to_string())

fig, ax = plt.subplots(figsize=(11, 5), facecolor=PALETTE["bg"])
cleaned = df_sil.dropna()
vals_s  = cleaned["Silhouette"].values
names_s = cleaned.index.tolist()
bar_colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(names_s)))
bars3 = ax.bar(names_s, vals_s, color=bar_colors, edgecolor=PALETTE["bg"], width=0.6)
for bar, v in zip(bars3, vals_s):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.003,
            f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=PALETTE["text"])
ax.set_xticks(range(len(names_s)))
ax.set_xticklabels(names_s, rotation=35, ha="right")
ax.set_ylim(0, max(vals_s) * 1.2)
ax.set_ylabel("Silhouette Score")
ax.set_title("Resumen General — Todos los Métodos de Clustering", fontsize=13)
estilo_oscuro(fig, [ax])
fig.tight_layout()
guardar(fig, "07_comparacion_clustering.png")

# ==============================================================
# PARTE 2 — RE-EVALUACIÓN DE ETIQUETAS
# ==============================================================

print("\n" + "━"*60)
print("  PARTE 2 — RE-EVALUACIÓN DE ETIQUETAS")
print("━"*60)

km2 = KMeans(n_clusters=2, random_state=SEED, n_init=10)
labels_km2 = km2.fit_predict(X_proc)

def alinear_clusters(y_true, y_pred):
    from itertools import permutations
    best_acc, best_map = 0, None
    for perm in permutations(range(2)):
        mapping = {old: new for old, new in enumerate(perm)}
        mapped = np.vectorize(mapping.get)(y_pred)
        acc = (y_true == mapped).mean()
        if acc > best_acc:
            best_acc, best_map = acc, mapping
    return np.vectorize(best_map.get)(y_pred), best_acc

km2_aligned, acc_align = alinear_clusters(y_original, labels_km2)
ari = adjusted_rand_score(y_original, km2_aligned)
print(f"\nAlineación KMeans(k=2) vs target: {acc_align:.2%}")
print(f"Adjusted Rand Index: {ari:.4f}")

desacuerdo = y_original != km2_aligned
print(f"\nMuestras en desacuerdo: {desacuerdo.sum()} / {len(y_original)} ({desacuerdo.mean():.1%})")

distancias = km2.transform(X_proc)
dist_propia = np.array([distancias[i, labels_km2[i]] for i in range(len(labels_km2))])

candidatos  = np.where(desacuerdo)[0]
orden_cand  = candidatos[np.argsort(dist_propia[candidatos])[::-1]]
max_cambios = int(0.30 * len(y_original))
n_cambios   = min(len(candidatos), max_cambios)

y_relabeled = y_original.copy()
idx_cambiados = orden_cand[:n_cambios]
y_relabeled[idx_cambiados] = 1 - y_relabeled[idx_cambiados]

print(f"\nCambios realizados: {n_cambios}  ({n_cambios/len(y_original):.1%})")
print(f"Distribución original    : {dict(zip(*np.unique(y_original, return_counts=True)))}")
print(f"Distribución re-etiquetada: {dict(zip(*np.unique(y_relabeled, return_counts=True)))}")

# Visualización — scatter side-by-side con diferencia resaltada
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 6), facecolor=PALETTE["bg"])

def scatter_labels(ax, labels, X2, title, idx_highlight=None):
    colors = np.where(labels == 1, PALETTE["accent1"], PALETTE["accent2"])
    ax.scatter(X2[:, 0], X2[:, 1], c=colors, s=35, alpha=0.75, edgecolors="none")
    if idx_highlight is not None and len(idx_highlight):
        ax.scatter(X2[idx_highlight, 0], X2[idx_highlight, 1],
                   facecolors="none", edgecolors=PALETTE["accent4"],
                   s=90, linewidths=1.8, label=f"Re-etiquetadas ({len(idx_highlight)})")
        ax.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"], fontsize=9)
    patch_si = Patch(color=PALETTE["accent1"], label="Desertó: Sí")
    patch_no = Patch(color=PALETTE["accent2"], label="Desertó: No")
    ax.legend(handles=[patch_si, patch_no] + (ax.get_legend_handles_labels()[0][2:]
                if idx_highlight is not None else []),
              facecolor=PALETTE["panel"], labelcolor=PALETTE["text"], fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("CP1"); ax.set_ylabel("CP2")

scatter_labels(ax_a, y_original,  X_pca, "Etiquetas Originales")
scatter_labels(ax_b, y_relabeled, X_pca, "Etiquetas Re-evaluadas", idx_cambiados)
estilo_oscuro(fig, [ax_a, ax_b])
fig.suptitle("Re-evaluación de Etiquetas — KMeans k=2", color=PALETTE["text"], fontsize=13)
fig.tight_layout()
guardar(fig, "08_reetiquetado.png")

# ==============================================================
# PARTE 3 — MODELOS SUPERVISADOS
# ==============================================================

print("\n" + "━"*60)
print("  PARTE 3 — MODELOS SUPERVISADOS (RE-ETIQUETADO)")
print("━"*60)

def construir_pipe(clf):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    clf)
    ])

def dividir(X, y, rs=SEED):
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.40, stratify=y, random_state=rs)
    Xv,  Xte,  yv,  yte  = train_test_split(Xtmp, ytmp, test_size=0.50, stratify=ytmp, random_state=rs)
    return Xtr, Xv, Xte, ytr, yv, yte

def evaluar(modelo, X, y, nombre, prefijo=""):
    yp = modelo.predict(X)
    yprob = modelo.predict_proba(X)[:, 1] if hasattr(modelo, "predict_proba") else None
    acc  = accuracy_score(y, yp)
    prec = precision_score(y, yp, zero_division=0)
    rec  = recall_score(y, yp, zero_division=0)
    f1   = f1_score(y, yp, zero_division=0)
    auc  = roc_auc_score(y, yprob) if yprob is not None else np.nan
    print(f"  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

    cm = confusion_matrix(y, yp)
    fig = plt.figure(figsize=(13, 5), facecolor=PALETTE["bg"])
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    sns.heatmap(cm, annot=True, fmt="d", ax=ax1,
                cmap="YlOrRd", linewidths=2, linecolor=PALETTE["bg"],
                cbar_kws={"shrink": 0.8},
                annot_kws={"size": 14, "color": "white"})
    ax1.set_title(f"Matriz de Confusión\n{nombre}", fontsize=11)
    ax1.set_ylabel("Real"); ax1.set_xlabel("Predicción")
    ax1.set_xticklabels(["No Desertó", "Desertó"])
    ax1.set_yticklabels(["No Desertó", "Desertó"], rotation=0)

    if yprob is not None:
        fpr, tpr, _ = roc_curve(y, yprob)
        ax2.fill_between(fpr, tpr, alpha=0.25, color=PALETTE["accent4"])
        ax2.plot(fpr, tpr, color=PALETTE["accent4"], linewidth=2.5, label=f"AUC={auc:.3f}")
        ax2.plot([0,1],[0,1], "--", color="gray", linewidth=1.2, label="Aleatorio")
        ax2.set_title(f"Curva ROC — {nombre}", fontsize=11)
        ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
        ax2.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"])
    else:
        ax2.axis("off")

    estilo_oscuro(fig, [ax1, ax2])
    fig.tight_layout()
    guardar(fig, f"{prefijo}_cm_roc.png")
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "AUC": auc}

# Splits
Xtr_r, Xv_r, Xte_r, ytr_r, yv_r, yte_r = dividir(X_proc, y_relabeled)
Xtr_o, Xv_o, Xte_o, ytr_o, yv_o, yte_o = dividir(X_proc, y_original)
print(f"\nTrain={len(Xtr_r)} | Val={len(Xv_r)} | Test={len(Xte_r)}")

# ─── Árbol de Decisión ────────────────────────────────────
print("\n--- Árbol de Decisión (re-etiquetado) ---")
dt = construir_pipe(DecisionTreeClassifier(max_depth=5, min_samples_leaf=8, random_state=SEED))
dt.fit(Xtr_r, ytr_r)
print("  [Train]"); res_dt_tr_r = evaluar(dt, Xtr_r, ytr_r, "DT Train  [relabeled]", "09_dt_train_r")
print("  [Val]");   res_dt_v_r  = evaluar(dt, Xv_r,  yv_r,  "DT Val    [relabeled]", "10_dt_val_r")
print("  [Test]");  res_dt_te_r = evaluar(dt, Xte_r, yte_r, "DT Test   [relabeled]", "11_dt_test_r")

# Árbol visual
feat_names = ["Promedio", "Materias_Perdidas", "Becado_num"]
fig_tree, ax_t = plt.subplots(figsize=(20, 9), facecolor=PALETTE["bg"])
ax_t.set_facecolor(PALETTE["panel"])
plot_tree(dt.named_steps["clf"], feature_names=feat_names,
          class_names=["No Desertó", "Desertó"],
          filled=True, max_depth=3, ax=ax_t,
          impurity=True, rounded=True, fontsize=9)
ax_t.set_title("Árbol de Decisión — Deserción (re-etiquetado)", color=PALETTE["text"], fontsize=13)
guardar(fig_tree, "12_arbol_decision.png")

# ─── Regresión Logística ──────────────────────────────────
print("\n--- Regresión Logística (re-etiquetado) ---")
lr = construir_pipe(LogisticRegression(max_iter=1000, C=1.0, random_state=SEED))
lr.fit(Xtr_r, ytr_r)
print("  [Train]"); res_lr_tr_r = evaluar(lr, Xtr_r, ytr_r, "LR Train  [relabeled]", "13_lr_train_r")
print("  [Val]");   res_lr_v_r  = evaluar(lr, Xv_r,  yv_r,  "LR Val    [relabeled]", "14_lr_val_r")
print("  [Test]");  res_lr_te_r = evaluar(lr, Xte_r, yte_r, "LR Test   [relabeled]", "15_lr_test_r")

# ─── Regresión Lineal (umbral 0.5) ────────────────────────
print("\n--- Regresión Lineal → umbral 0.5 (re-etiquetado) ---")

linreg = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
linreg.fit(Xtr_r, ytr_r)

def evaluar_linreg(pipe, X, y, nombre, prefijo=""):
    y_cont = pipe.predict(X)
    yp     = (y_cont >= 0.5).astype(int)
    acc    = accuracy_score(y, yp)
    prec   = precision_score(y, yp, zero_division=0)
    rec    = recall_score(y, yp, zero_division=0)
    f1     = f1_score(y, yp, zero_division=0)
    auc    = roc_auc_score(y, y_cont) if len(np.unique(y)) == 2 else np.nan
    mse    = mean_squared_error(y, y_cont)
    r2     = r2_score(y, y_cont)
    print(f"  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}  MSE={mse:.4f}  R²={r2:.4f}")

    fig = plt.figure(figsize=(13, 5), facecolor=PALETTE["bg"])
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    cm = confusion_matrix(y, yp)
    sns.heatmap(cm, annot=True, fmt="d", ax=ax1, cmap="YlOrRd",
                linewidths=2, linecolor=PALETTE["bg"],
                annot_kws={"size": 14, "color": "white"})
    ax1.set_title(f"Matriz de Confusión\n{nombre}", fontsize=11)
    ax1.set_ylabel("Real"); ax1.set_xlabel("Predicción")
    ax1.set_xticklabels(["No Desertó", "Desertó"])
    ax1.set_yticklabels(["No Desertó", "Desertó"], rotation=0)

    fpr, tpr, _ = roc_curve(y, y_cont)
    ax2.fill_between(fpr, tpr, alpha=0.2, color=PALETTE["accent3"])
    ax2.plot(fpr, tpr, color=PALETTE["accent3"], linewidth=2.5, label=f"AUC={auc:.3f}")
    ax2.plot([0,1],[0,1], "--", color="gray", linewidth=1.2)
    ax2.set_title(f"Curva ROC — {nombre}", fontsize=11)
    ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
    ax2.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"])

    estilo_oscuro(fig, [ax1, ax2])
    fig.tight_layout()
    guardar(fig, f"{prefijo}_cm_roc.png")
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1,
            "AUC": auc, "MSE": mse, "R²": r2}

print("  [Train]"); res_lin_tr_r = evaluar_linreg(linreg, Xtr_r, ytr_r, "LinReg Train [relabeled]", "16_linreg_train_r")
print("  [Val]");   res_lin_v_r  = evaluar_linreg(linreg, Xv_r,  yv_r,  "LinReg Val   [relabeled]", "17_linreg_val_r")
print("  [Test]");  res_lin_te_r = evaluar_linreg(linreg, Xte_r, yte_r, "LinReg Test  [relabeled]", "18_linreg_test_r")

# ==============================================================
# PARTE 4 — COMPARACIÓN ORIGINAL vs RE-ETIQUETADO
# ==============================================================

print("\n" + "━"*60)
print("  PARTE 4 — COMPARACIÓN: ORIGINAL vs RE-ETIQUETADO")
print("━"*60)

print("\n--- DT original ---")
dt_o = construir_pipe(DecisionTreeClassifier(max_depth=5, min_samples_leaf=8, random_state=SEED))
dt_o.fit(Xtr_o, ytr_o)
print("  [Test]"); res_dt_te_o = evaluar(dt_o, Xte_o, yte_o, "DT Test [original]", "19_dt_test_o")

print("\n--- LR original ---")
lr_o = construir_pipe(LogisticRegression(max_iter=1000, C=1.0, random_state=SEED))
lr_o.fit(Xtr_o, ytr_o)
print("  [Test]"); res_lr_te_o = evaluar(lr_o, Xte_o, yte_o, "LR Test [original]", "20_lr_test_o")

print("\n--- LinReg original ---")
lin_o = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
lin_o.fit(Xtr_o, ytr_o)
print("  [Test]"); res_lin_te_o = evaluar_linreg(lin_o, Xte_o, yte_o, "LinReg Test [original]", "21_linreg_test_o")

# --- Tabla comparativa ---
df_comp = pd.DataFrame({
    "DT — Original":       res_dt_te_o,
    "DT — Re-etiquetado":  res_dt_te_r,
    "LR — Original":       res_lr_te_o,
    "LR — Re-etiquetado":  res_lr_te_r,
    "LinReg — Original":   res_lin_te_o,
    "LinReg — Re-etiquetado": res_lin_te_r,
}).T

print("\n--- Tabla comparativa (Test) ---")
print(df_comp.to_string())
df_comp.to_csv(OUTDIR / "comparacion_modelos.csv")

# Gráfica comparativa — estilo "grouped bar" oscuro
metricas_plot = ["Accuracy", "F1", "AUC"]
fig_c = plt.figure(figsize=(16, 6), facecolor=PALETTE["bg"])
gs_c  = gridspec.GridSpec(1, 3, figure=fig_c, wspace=0.4)

grupo_orig  = [k for k in df_comp.index if "Original"      in k]
grupo_rela  = [k for k in df_comp.index if "Re-etiquetado" in k]
x_pos = np.arange(3)  # DT, LR, LinReg

for idx_m, met in enumerate(metricas_plot):
    ax_c = fig_c.add_subplot(gs_c[idx_m])
    vals_o = [df_comp.loc[k, met] for k in grupo_orig  if not np.isnan(df_comp.loc[k, met])]
    vals_r = [df_comp.loc[k, met] for k in grupo_rela  if not np.isnan(df_comp.loc[k, met])]
    width = 0.35
    bars_o = ax_c.bar(x_pos - width/2, vals_o, width, color=PALETTE["accent2"],
                      edgecolor=PALETTE["bg"], label="Original")
    bars_r = ax_c.bar(x_pos + width/2, vals_r, width, color=PALETTE["accent1"],
                      edgecolor=PALETTE["bg"], label="Re-etiquetado")
    for bar, v in zip(list(bars_o) + list(bars_r), vals_o + vals_r):
        ax_c.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                  f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=PALETTE["text"])
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(["DT", "LR", "LinReg"])
    ax_c.set_ylim(0, 1.15)
    ax_c.set_title(f"Comparación — {met}", fontsize=11)
    ax_c.set_ylabel(met)
    ax_c.legend(facecolor=PALETTE["panel"], labelcolor=PALETTE["text"], fontsize=8)
    estilo_oscuro(fig_c, [ax_c])

fig_c.suptitle("Original vs Re-etiquetado — Test Set", color=PALETTE["text"], fontsize=13, y=1.02)
fig_c.tight_layout()
guardar(fig_c, "22_comparacion_modelos.png")

# ==============================================================
# RESUMEN FINAL
# ==============================================================

print("\n" + "━"*60)
print("  RESUMEN FINAL")
print("━"*60)
print(f"  Dataset: {data.shape[0]} estudiantes, {X_raw.shape[1]} features")
print(f"  Target: Desertó  (No={( y_original==0).sum()} | Sí={(y_original==1).sum()})")
print(f"  Muestras re-etiquetadas: {n_cambios} ({n_cambios/len(y_original):.1%})")
print(f"\n  Silhouette por método:")
for k, v in df_sil.dropna().iterrows():
    print(f"    {k:25s}: {v['Silhouette']:.4f}")
print(f"\n  Test set — comparación:")
print(df_comp[["Accuracy","F1","AUC"]].to_string())
print(f"\n  Gráficas guardadas en: {OUTDIR.resolve()}")
print("\n  [FIN]\n")
