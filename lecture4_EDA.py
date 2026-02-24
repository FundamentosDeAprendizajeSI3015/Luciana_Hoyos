# ==========================================================
# ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# Proyecto: Predicción de Deserción Estudiantil
# Universidad EAFIT - Fundamentos de Aprendizaje Automático
# ==========================================================

import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Visualizaciones
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# ==========================================================
# CONFIGURACIÓN
# ==========================================================

DATA_CSV = "dataset_desercion_estudiantes.csv"
OUTDIR = Path("./eda_output")
OUTDIR.mkdir(parents=True, exist_ok=True)

TARGET = "Desertó"

NUM_COLS = [
    "Promedio",
    "Materias_Perdidas"
]

CAT_COLS = [
    "Becado"
]

# ==========================================================
# CARGA DE DATOS
# ==========================================================

print("\n" + "="*70)
print("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
print("Proyecto: Predicción de Deserción Estudiantil")
print("="*70)

print("\n=== 📊 CARGANDO DATASET ===")
df = pd.read_csv(DATA_CSV)
print(f"Shape: {df.shape}")
print(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")

print(f"\n--- Primeras 5 filas ---")
print(df.head())

print(f"\n--- Información del dataset ---")
print(df.info())

print(f"\n--- Valores nulos ---")
nulos = df.isna().sum()
print(nulos)
if nulos.sum() == 0:
    print("✓ No hay valores nulos en el dataset")

print(f"\n--- Distribución de la variable objetivo '{TARGET}' ---")
print(df[TARGET].value_counts())
print("\nProporción:")
print(df[TARGET].value_counts(normalize=True))

# Guardar descripción básica
df.describe().to_csv(OUTDIR / "00_descripcion_basica.csv")
print(f"\n✓ Guardado: {OUTDIR}/00_descripcion_basica.csv")

# ==========================================================
# 1. MEDIDAS DE TENDENCIA CENTRAL
# ==========================================================

print("\n" + "="*70)
print("1. MEDIDAS DE TENDENCIA CENTRAL")
print("="*70)

# Convertir categóricas a numéricas para análisis
df_numeric = df.copy()
df_numeric['Becado_num'] = df_numeric['Becado'].map({'Sí': 1, 'No': 0})
df_numeric['Desertó_num'] = df_numeric['Desertó'].map({'Sí': 1, 'No': 0})

# Numéricas
print("\n--- Variables Numéricas ---")
media_num = df[NUM_COLS].mean()
mediana_num = df[NUM_COLS].median()
moda_num = df[NUM_COLS].mode().iloc[0]

tendencia_numericas = pd.DataFrame({
    "Media": media_num,
    "Mediana": mediana_num,
    "Moda": moda_num
})
print(tendencia_numericas)
tendencia_numericas.to_csv(OUTDIR / "01_tendencia_central_numericas.csv")
print(f"✓ Guardado: {OUTDIR}/01_tendencia_central_numericas.csv")

# Categóricas - Modas
print("\n--- Variables Categóricas (Modas) ---")
moda_cat = {
    'Becado': df['Becado'].mode().tolist(),
    'Desertó': df['Desertó'].mode().tolist()
}
print(f"Becado: {moda_cat['Becado']}")
print(f"Desertó: {moda_cat['Desertó']}")

with open(OUTDIR / "01_moda_categoricas.json", "w", encoding="utf-8") as f:
    json.dump(moda_cat, f, indent=2, ensure_ascii=False)
print(f"✓ Guardado: {OUTDIR}/01_moda_categoricas.json")

# Proporciones de variables categóricas
print("\n--- Proporciones de Variables Categóricas ---")
prop_becado = df['Becado'].value_counts(normalize=True)
prop_deserto = df['Desertó'].value_counts(normalize=True)

print("\nBecado:")
print(prop_becado)
print("\nDesertó:")
print(prop_deserto)

proporciones = {
    'Becado': {str(k): float(v) for k, v in prop_becado.to_dict().items()},
    'Desertó': {str(k): float(v) for k, v in prop_deserto.to_dict().items()}
}

with open(OUTDIR / "01_proporciones_categoricas.json", "w", encoding="utf-8") as f:
    json.dump(proporciones, f, indent=2, ensure_ascii=False)
print(f"✓ Guardado: {OUTDIR}/01_proporciones_categoricas.json")

# ==========================================================
# 2. CUARTILES E IQR
# ==========================================================

print("\n" + "="*70)
print("2. CUARTILES E RANGO INTERCUARTÍLICO (IQR)")
print("="*70)

iqr_results = {}

for col in NUM_COLS:
    Q1 = df[col].quantile(0.25)
    Q2 = df[col].quantile(0.50)  # Mediana
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    iqr_results[col] = {
        "Q1 (25%)": float(Q1),
        "Q2 (50% - Mediana)": float(Q2),
        "Q3 (75%)": float(Q3),
        "IQR": float(IQR),
        "Límite_Inferior_Outliers": float(Q1 - 1.5 * IQR),
        "Límite_Superior_Outliers": float(Q3 + 1.5 * IQR)
    }
    
    print(f"\n--- {col} ---")
    print(f"Q1 (25%): {Q1:.2f}")
    print(f"Q2 (50%): {Q2:.2f}")
    print(f"Q3 (75%): {Q3:.2f}")
    print(f"IQR: {IQR:.2f}")
    print(f"Límite outliers: [{Q1 - 1.5*IQR:.2f}, {Q3 + 1.5*IQR:.2f}]")

with open(OUTDIR / "02_iqr_results.json", "w", encoding="utf-8") as f:
    json.dump(iqr_results, f, indent=2, ensure_ascii=False)
print(f"\n✓ Guardado: {OUTDIR}/02_iqr_results.json")

# ==========================================================
# 3. PERCENTILES
# ==========================================================

print("\n" + "="*70)
print("3. PERCENTILES")
print("="*70)

percentiles = {}

for col in NUM_COLS:
    percentiles[col] = {
        "P10 (10%)": float(np.percentile(df[col], 10)),
        "P25 (25%)": float(np.percentile(df[col], 25)),
        "P50 (50%)": float(np.percentile(df[col], 50)),
        "P75 (75%)": float(np.percentile(df[col], 75)),
        "P90 (90%)": float(np.percentile(df[col], 90))
    }
    
    print(f"\n--- {col} ---")
    print(f"P10: {percentiles[col]['P10 (10%)']:.2f}")
    print(f"P25: {percentiles[col]['P25 (25%)']:.2f}")
    print(f"P50: {percentiles[col]['P50 (50%)']:.2f}")
    print(f"P75: {percentiles[col]['P75 (75%)']:.2f}")
    print(f"P90: {percentiles[col]['P90 (90%)']:.2f}")

with open(OUTDIR / "03_percentiles.json", "w", encoding="utf-8") as f:
    json.dump(percentiles, f, indent=2, ensure_ascii=False)
print(f"\n✓ Guardado: {OUTDIR}/03_percentiles.json")

# ==========================================================
# 4. CORRELACIONES
# ==========================================================

print("\n" + "="*70)
print("4. ANÁLISIS DE CORRELACIÓN")
print("="*70)

# Matriz de correlación
corr_matrix = df_numeric[NUM_COLS + ['Becado_num', 'Desertó_num']].corr()
print("\n--- Matriz de Correlación ---")
print(corr_matrix)

# Visualización: Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", center=0,
            cbar_kws={'label': 'Correlación'})
plt.title("Mapa de Calor - Correlaciones (Pearson)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTDIR / "04_heatmap_correlacion.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"\n✓ Guardado: {OUTDIR}/04_heatmap_correlacion.png")

# Correlación Pearson vs Spearman
print("\n--- Correlación con Variable Objetivo (Deserción) ---")
correlation_stats = {}

for col in NUM_COLS + ['Becado_num']:
    pearson = df_numeric[col].corr(df_numeric['Desertó_num'], method="pearson")
    spearman = df_numeric[col].corr(df_numeric['Desertó_num'], method="spearman")
    
    correlation_stats[col] = {
        "pearson": float(pearson),
        "spearman": float(spearman)
    }
    
    print(f"\n{col}:")
    print(f"  Pearson:  {pearson:+.3f}")
    print(f"  Spearman: {spearman:+.3f}")

with open(OUTDIR / "04_correlation_stats.json", "w", encoding="utf-8") as f:
    json.dump(correlation_stats, f, indent=2, ensure_ascii=False)
print(f"\n✓ Guardado: {OUTDIR}/04_correlation_stats.json")

# ==========================================================
# 5. TABLAS PIVOTE
# ==========================================================

print("\n" + "="*70)
print("5. TABLAS PIVOTE (AGREGACIONES)")
print("="*70)

# Promedio académico por Beca y Deserción
print("\n--- Promedio Académico por Beca y Deserción ---")
pivot_promedio = df.pivot_table(
    index="Becado",
    columns="Desertó",
    values="Promedio",
    aggfunc="mean"
)
print(pivot_promedio)
pivot_promedio.to_csv(OUTDIR / "05_pivot_promedio_beca_desercion.csv")
print(f"✓ Guardado: {OUTDIR}/05_pivot_promedio_beca_desercion.csv")

# Materias perdidas por Beca y Deserción
print("\n--- Materias Perdidas (promedio) por Beca y Deserción ---")
pivot_materias = df.pivot_table(
    index="Becado",
    columns="Desertó",
    values="Materias_Perdidas",
    aggfunc="mean"
)
print(pivot_materias)
pivot_materias.to_csv(OUTDIR / "05_pivot_materias_beca_desercion.csv")
print(f"✓ Guardado: {OUTDIR}/05_pivot_materias_beca_desercion.csv")

# Conteo por combinación
print("\n--- Conteo de Estudiantes por Beca y Deserción ---")
pivot_count = df.pivot_table(
    index="Becado",
    columns="Desertó",
    values="ID",
    aggfunc="count"
)
print(pivot_count)
pivot_count.to_csv(OUTDIR / "05_pivot_count_beca_desercion.csv")
print(f"✓ Guardado: {OUTDIR}/05_pivot_count_beca_desercion.csv")

# ==========================================================
# 6. VISUALIZACIONES
# ==========================================================

print("\n" + "="*70)
print("6. GENERANDO VISUALIZACIONES")
print("="*70)

# 6.1 Histogramas por variable
print("\n--- Histogramas de distribución ---")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, col in enumerate(NUM_COLS):
    row = idx // 2
    col_idx = idx % 2
    
    axes[row, col_idx].hist(
        df[df['Desertó'] == 'No'][col], 
        alpha=0.6, 
        label='No Desertó', 
        bins=20, 
        color='#3498db', 
        edgecolor='black'
    )
    axes[row, col_idx].hist(
        df[df['Desertó'] == 'Sí'][col], 
        alpha=0.6, 
        label='Desertó', 
        bins=20, 
        color='#e74c3c', 
        edgecolor='black'
    )
    axes[row, col_idx].set_xlabel(col, fontsize=11)
    axes[row, col_idx].set_ylabel('Frecuencia', fontsize=11)
    axes[row, col_idx].set_title(f'Distribución: {col}', fontsize=12, fontweight='bold')
    axes[row, col_idx].legend()
    axes[row, col_idx].grid(True, alpha=0.3)

# Gráfico de barras para Becado
axes[1, 1].clear()
df_grouped = df.groupby(['Becado', 'Desertó']).size().unstack(fill_value=0)
df_grouped.plot(kind='bar', ax=axes[1, 1], color=['#3498db', '#e74c3c'], edgecolor='black')
axes[1, 1].set_title('Deserción por Becado', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Becado', fontsize=11)
axes[1, 1].set_ylabel('Cantidad de Estudiantes', fontsize=11)
axes[1, 1].legend(title='Desertó')
axes[1, 1].grid(True, alpha=0.3, axis='y')
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig(OUTDIR / "06_histogramas_distribuciones.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Guardado: {OUTDIR}/06_histogramas_distribuciones.png")

# 6.2 Boxplots
print("\n--- Boxplots por deserción ---")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, col in enumerate(NUM_COLS):
    df.boxplot(column=col, by='Desertó', ax=axes[idx], patch_artist=True)
    axes[idx].set_title(f'{col} por Deserción', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Desertó', fontsize=11)
    axes[idx].set_ylabel(col, fontsize=11)
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('')
plt.tight_layout()
plt.savefig(OUTDIR / "06_boxplots_por_desercion.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Guardado: {OUTDIR}/06_boxplots_por_desercion.png")

# 6.3 Scatter plot: Promedio vs Materias Perdidas
print("\n--- Scatter plot: Promedio vs Materias Perdidas ---")
fig, ax = plt.subplots(figsize=(10, 6))
colors = {'No': '#3498db', 'Sí': '#e74c3c'}
markers = {'No': 'No', 'Sí': 'Sí'}

for desertion, group in df.groupby('Desertó'):
    ax.scatter(
        group['Promedio'], 
        group['Materias_Perdidas'], 
        label=f'Desertó: {desertion}', 
        c=colors[desertion], 
        alpha=0.6, 
        edgecolors='black',
        s=50
    )

ax.set_xlabel('Promedio Académico', fontsize=12)
ax.set_ylabel('Materias Perdidas', fontsize=12)
ax.set_title('Promedio vs Materias Perdidas por Deserción', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTDIR / "06_scatter_promedio_materias.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Guardado: {OUTDIR}/06_scatter_promedio_materias.png")

# 6.4 Gráficos de barras para proporciones
print("\n--- Gráficos de barras de proporciones ---")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Becado
becado_counts = df['Becado'].value_counts()
axes[0].bar(becado_counts.index, becado_counts.values, color=['#3498db', '#95a5a6'], edgecolor='black')
axes[0].set_title('Distribución: Becado', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Cantidad de Estudiantes', fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(becado_counts.values):
    axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')

# Deserción
deserto_counts = df['Desertó'].value_counts()
axes[1].bar(deserto_counts.index, deserto_counts.values, color=['#3498db', '#e74c3c'], edgecolor='black')
axes[1].set_title('Distribución: Deserción', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Cantidad de Estudiantes', fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(deserto_counts.values):
    axes[1].text(i, v + 5, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTDIR / "06_barras_proporciones.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Guardado: {OUTDIR}/06_barras_proporciones.png")

# 6.5 Distribución por Beca y Deserción (stacked bar)
print("\n--- Gráfico stacked: Deserción por Beca ---")
pivot_for_plot = df.groupby(['Becado', 'Desertó']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(8, 6))
pivot_for_plot.plot(kind='bar', stacked=True, ax=ax, color=['#3498db', '#e74c3c'], edgecolor='black')
ax.set_title('Deserción por Becado (Stacked)', fontsize=14, fontweight='bold')
ax.set_xlabel('Becado', fontsize=12)
ax.set_ylabel('Cantidad de Estudiantes', fontsize=12)
ax.legend(title='Desertó', loc='upper right')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(True, alpha=0.3, axis='y')

# Añadir valores en las barras
for container in ax.containers:
    ax.bar_label(container, label_type='center', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTDIR / "06_stacked_desercion_beca.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Guardado: {OUTDIR}/06_stacked_desercion_beca.png")

# ==========================================================
# 7. RESUMEN ESTADÍSTICO POR CLASE
# ==========================================================

print("\n" + "="*70)
print("7. RESUMEN ESTADÍSTICO POR CLASE")
print("="*70)

print("\n--- Estadísticas por Deserción ---")
resumen_por_clase = df.groupby('Desertó')[NUM_COLS].describe()
print(resumen_por_clase)
resumen_por_clase.to_csv(OUTDIR / "07_resumen_estadistico_por_clase.csv")
print(f"\n✓ Guardado: {OUTDIR}/07_resumen_estadistico_por_clase.csv")

# Comparación específica
print("\n--- Comparación de Medias por Deserción ---")
comparacion_medias = df.groupby('Desertó')[NUM_COLS + ['Becado_num']].mean()
print(comparacion_medias)
comparacion_medias.to_csv(OUTDIR / "07_comparacion_medias_por_clase.csv")
print(f"✓ Guardado: {OUTDIR}/07_comparacion_medias_por_clase.csv")

# ==========================================================
# 8. IDENTIFICACIÓN DE OUTLIERS
# ==========================================================

print("\n" + "="*70)
print("8. IDENTIFICACIÓN DE OUTLIERS")
print("="*70)

outliers_info = {}

for col in NUM_COLS:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    num_outliers = len(outliers)
    pct_outliers = (num_outliers / len(df)) * 100
    
    outliers_info[col] = {
        "num_outliers": int(num_outliers),
        "porcentaje": float(pct_outliers),
        "limite_inferior": float(lower_bound),
        "limite_superior": float(upper_bound)
    }
    
    print(f"\n--- {col} ---")
    print(f"Outliers detectados: {num_outliers} ({pct_outliers:.2f}%)")
    print(f"Límites: [{lower_bound:.2f}, {upper_bound:.2f}]")

with open(OUTDIR / "08_outliers_info.json", "w", encoding="utf-8") as f:
    json.dump(outliers_info, f, indent=2, ensure_ascii=False)
print(f"\n✓ Guardado: {OUTDIR}/08_outliers_info.json")

# ==========================================================
# RESUMEN FINAL
# ==========================================================

print("\n" + "="*70)
print("✅ ANÁLISIS EXPLORATORIO COMPLETADO")
print("="*70)

print(f"\n📂 Todos los archivos guardados en: {OUTDIR}/")
print("\n📊 Archivos generados:")
print("\n  Estadísticas:")
print("     - 00_descripcion_basica.csv")
print("     - 01_tendencia_central_numericas.csv")
print("     - 01_moda_categoricas.json")
print("     - 01_proporciones_categoricas.json")
print("     - 02_iqr_results.json")
print("     - 03_percentiles.json")
print("     - 04_correlation_stats.json")
print("     - 05_pivot_promedio_beca_desercion.csv")
print("     - 05_pivot_materias_beca_desercion.csv")
print("     - 05_pivot_count_beca_desercion.csv")
print("     - 07_resumen_estadistico_por_clase.csv")
print("     - 07_comparacion_medias_por_clase.csv")
print("     - 08_outliers_info.json")
print("\n  Visualizaciones:")
print("     - 04_heatmap_correlacion.png")
print("     - 06_histogramas_distribuciones.png")
print("     - 06_boxplots_por_desercion.png")
print("     - 06_scatter_promedio_materias.png")
print("     - 06_barras_proporciones.png")
print("     - 06_stacked_desercion_beca.png")

print("\n" + "="*70)
print("📈 INSIGHTS CLAVE")
print("="*70)

# Calcular algunos insights
prom_desertores = df[df['Desertó'] == 'Sí']['Promedio'].mean()
prom_no_desertores = df[df['Desertó'] == 'No']['Promedio'].mean()
mat_desertores = df[df['Desertó'] == 'Sí']['Materias_Perdidas'].mean()
mat_no_desertores = df[df['Desertó'] == 'No']['Materias_Perdidas'].mean()
pct_becados_desertores = (df[df['Desertó'] == 'Sí']['Becado'] == 'Sí').sum() / len(df[df['Desertó'] == 'Sí']) * 100
pct_becados_no_desertores = (df[df['Desertó'] == 'No']['Becado'] == 'Sí').sum() / len(df[df['Desertó'] == 'No']) * 100

print(f"\n1. Promedio Académico:")
print(f"   - Desertores: {prom_desertores:.2f}")
print(f"   - No Desertores: {prom_no_desertores:.2f}")
print(f"   - Diferencia: {prom_no_desertores - prom_desertores:.2f} puntos")

print(f"\n2. Materias Perdidas:")
print(f"   - Desertores: {mat_desertores:.2f}")
print(f"   - No Desertores: {mat_no_desertores:.2f}")
print(f"   - Diferencia: {mat_desertores - mat_no_desertores:.2f} materias")

print(f"\n3. Efecto de la Beca:")
print(f"   - Desertores becados: {pct_becados_desertores:.1f}%")
print(f"   - No desertores becados: {pct_becados_no_desertores:.1f}%")
print(f"   - Diferencia: {pct_becados_no_desertores - pct_becados_desertores:.1f} p.p.")

print("\n💡 Conclusión:")
print("   Los desertores tienen promedios más bajos, pierden más materias")
print("   y tienen menor probabilidad de tener beca. Estos patrones sugieren")
print("   que un modelo de ML puede predecir la deserción exitosamente.")

print("\n" + "="*70)
