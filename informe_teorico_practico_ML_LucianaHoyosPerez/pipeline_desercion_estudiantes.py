# ==========================================================
# PROYECTO: Predicción de Deserción Estudiantil
# Universidad EAFIT - Fundamentos de Aprendizaje Automático
# ==========================================================

import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Visualizaciones
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# CONFIGURACIÓN GENERAL
# ==========================================================

DATA_CSV = "dataset_desercion_estudiantes.csv"
OUTDIR = Path("./data_output_desercion")
OUTDIR.mkdir(parents=True, exist_ok=True)

TARGET = "Desertó"

# Variables categóricas (binarias en nuestro caso)
CAT_COLS = [
    "Becado"
]

# Variables numéricas
NUM_COLS = [
    "Promedio",
    "Materias_Perdidas"
]

# ==========================================================
# 1️⃣ DEFINICIÓN DEL PROBLEMA
# ==========================================================

problem_definition = {
    "objetivo": "Predecir si un estudiante desertará (Sí/No)",
    "impacto": "Identificar tempranamente estudiantes en riesgo para aplicar intervenciones",
    "tipo_problema": "Clasificación binaria supervisada",
    "variables_numericas": NUM_COLS,
    "variables_categoricas": CAT_COLS,
    "algoritmo_propuesto": "XGBoost",
    "justificacion_algoritmo": "Maneja datos heterogéneos, desbalanceo de clases y relaciones no lineales",
    "metrica_principal": "Recall (>80%)",
    "justificacion_metrica": "Preferimos detectar todos los casos de deserción (minimizar Falsos Negativos)"
}

with open(OUTDIR / "definicion_problema.json", "w", encoding="utf-8") as f:
    json.dump(problem_definition, f, indent=2, ensure_ascii=False)

print("\n" + "="*70)
print("PROYECTO: PREDICCIÓN DE DESERCIÓN ESTUDIANTIL")
print("="*70)

# ==========================================================
# 2️⃣ RECOLECCIÓN DE DATOS
# ==========================================================

print("\n=== 📊 CARGANDO DATASET ===")
df = pd.read_csv(DATA_CSV)
print(f"Shape inicial: {df.shape}")
print(f"\nPrimeras 3 filas:")
print(df.head(3))
print(f"\nInformación del dataset:")
print(df.info())
print(f"\nValores nulos por columna:")
print(df.isna().sum())
print(f"\nDistribución de la variable objetivo '{TARGET}':")
print(df[TARGET].value_counts())
print(df[TARGET].value_counts(normalize=True))

# Guardar descripción básica
df.describe().to_csv(OUTDIR / "descripcion_basica.csv")

# ==========================================================
# 3️⃣ ANÁLISIS EXPLORATORIO COMPLETO
# ==========================================================

print("\n=== 🔍 ANÁLISIS EXPLORATORIO DE DATOS (EDA) ===")

# ---------------------------
# Convertir variables categóricas a numéricas temporalmente para análisis
# ---------------------------
df_numeric = df.copy()
df_numeric['Becado_num'] = df_numeric['Becado'].map({'Sí': 1, 'No': 0})
df_numeric['Desertó_num'] = df_numeric['Desertó'].map({'Sí': 1, 'No': 0})

# ---------------------------
# 3.1 Tendencia Central
# ---------------------------
print("\n--- Medidas de Tendencia Central ---")

# Numéricas
media_num = df[NUM_COLS].mean()
mediana_num = df[NUM_COLS].median()
moda_num = df[NUM_COLS].mode().iloc[0]

tendencia_numericas = pd.DataFrame({
    "Media": media_num,
    "Mediana": mediana_num,
    "Moda": moda_num
})
print("\nVariables Numéricas:")
print(tendencia_numericas)
tendencia_numericas.to_csv(OUTDIR / "tendencia_central_numericas.csv")

# Categóricas
moda_cat = {
    'Becado': df['Becado'].mode().tolist(),
    'Desertó': df['Desertó'].mode().tolist()
}
print(f"\nModas Categóricas:")
print(f"  Becado: {moda_cat['Becado']}")
print(f"  Desertó: {moda_cat['Desertó']}")

with open(OUTDIR / "moda_categoricas.json", "w", encoding="utf-8") as f:
    json.dump(moda_cat, f, indent=2, ensure_ascii=False)

# Proporciones de variables categóricas
prop_becado = df['Becado'].value_counts(normalize=True)
prop_deserto = df['Desertó'].value_counts(normalize=True)

proporciones = {
    'Becado': {str(k): float(v) for k, v in prop_becado.to_dict().items()},
    'Desertó': {str(k): float(v) for k, v in prop_deserto.to_dict().items()}
}

with open(OUTDIR / "proporciones_categoricas.json", "w", encoding="utf-8") as f:
    json.dump(proporciones, f, indent=2, ensure_ascii=False)

# ---------------------------
# 3.2 Cuartiles e IQR
# ---------------------------
print("\n--- Cuartiles e IQR (Rango Intercuartílico) ---")

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
    
    print(f"\n{col}:")
    print(f"  Q1 (25%): {Q1:.2f}")
    print(f"  Q2 (50%): {Q2:.2f}")
    print(f"  Q3 (75%): {Q3:.2f}")
    print(f"  IQR: {IQR:.2f}")
    print(f"  Límite outliers: [{Q1 - 1.5*IQR:.2f}, {Q3 + 1.5*IQR:.2f}]")

with open(OUTDIR / "iqr_results.json", "w", encoding="utf-8") as f:
    json.dump(iqr_results, f, indent=2, ensure_ascii=False)

# ---------------------------
# 3.3 Percentiles
# ---------------------------
print("\n--- Percentiles ---")

percentiles = {}

for col in NUM_COLS:
    percentiles[col] = {
        "P10 (10%)": float(np.percentile(df[col], 10)),
        "P25 (25%)": float(np.percentile(df[col], 25)),
        "P50 (50%)": float(np.percentile(df[col], 50)),
        "P75 (75%)": float(np.percentile(df[col], 75)),
        "P90 (90%)": float(np.percentile(df[col], 90))
    }
    
    print(f"\n{col}:")
    print(f"  P10: {percentiles[col]['P10 (10%)']:.2f}")
    print(f"  P50: {percentiles[col]['P50 (50%)']:.2f}")
    print(f"  P90: {percentiles[col]['P90 (90%)']:.2f}")

with open(OUTDIR / "percentiles.json", "w", encoding="utf-8") as f:
    json.dump(percentiles, f, indent=2, ensure_ascii=False)

# ---------------------------
# 3.4 Correlación
# ---------------------------
print("\n--- Análisis de Correlación ---")

# Correlación con todas las variables numéricas
corr_matrix = df_numeric[NUM_COLS + ['Becado_num', 'Desertó_num']].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", center=0)
plt.title("Mapa de Calor - Correlaciones (Pearson)")
plt.tight_layout()
plt.savefig(OUTDIR / "heatmap_correlacion.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Guardado: heatmap_correlacion.png")

# Correlación Pearson vs Spearman
correlation_stats = {}

for col in NUM_COLS + ['Becado_num']:
    correlation_stats[col] = {
        "pearson": float(df_numeric[col].corr(df_numeric['Desertó_num'], method="pearson")),
        "spearman": float(df_numeric[col].corr(df_numeric['Desertó_num'], method="spearman"))
    }
    
    print(f"\n{col} vs Deserción:")
    print(f"  Pearson:  {correlation_stats[col]['pearson']:.3f}")
    print(f"  Spearman: {correlation_stats[col]['spearman']:.3f}")

with open(OUTDIR / "correlation_stats.json", "w", encoding="utf-8") as f:
    json.dump(correlation_stats, f, indent=2, ensure_ascii=False)

# ---------------------------
# 3.5 Pivot Tables
# ---------------------------
print("\n--- Tablas Pivote ---")

# Promedio académico por Beca y Deserción
pivot_promedio = df.pivot_table(
    index="Becado",
    columns="Desertó",
    values="Promedio",
    aggfunc="mean"
)
print("\nPromedio Académico por Beca y Deserción:")
print(pivot_promedio)
pivot_promedio.to_csv(OUTDIR / "pivot_promedio_beca_desercion.csv")

# Materias perdidas por Beca y Deserción
pivot_materias = df.pivot_table(
    index="Becado",
    columns="Desertó",
    values="Materias_Perdidas",
    aggfunc="mean"
)
print("\nMaterias Perdidas (promedio) por Beca y Deserción:")
print(pivot_materias)
pivot_materias.to_csv(OUTDIR / "pivot_materias_beca_desercion.csv")

# Conteo por combinación
pivot_count = df.pivot_table(
    index="Becado",
    columns="Desertó",
    values="ID",
    aggfunc="count"
)
print("\nConteo de estudiantes por Beca y Deserción:")
print(pivot_count)
pivot_count.to_csv(OUTDIR / "pivot_count_beca_desercion.csv")

# ---------------------------
# 3.6 Distribuciones
# ---------------------------
print("\n--- Visualizaciones de Distribución ---")

# Histogramas
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, col in enumerate(NUM_COLS):
    row = idx // 2
    col_idx = idx % 2
    
    axes[row, col_idx].hist(df[df['Desertó'] == 'No'][col], alpha=0.6, label='No Desertó', bins=20, color='blue', edgecolor='black')
    axes[row, col_idx].hist(df[df['Desertó'] == 'Sí'][col], alpha=0.6, label='Desertó', bins=20, color='red', edgecolor='black')
    axes[row, col_idx].set_xlabel(col, fontsize=11)
    axes[row, col_idx].set_ylabel('Frecuencia', fontsize=11)
    axes[row, col_idx].set_title(f'Distribución: {col}', fontsize=12, fontweight='bold')
    axes[row, col_idx].legend()
    axes[row, col_idx].grid(True, alpha=0.3)

# Gráfico de barras para Becado
axes[1, 1].clear()
df_grouped = df.groupby(['Becado', 'Desertó']).size().unstack(fill_value=0)
df_grouped.plot(kind='bar', ax=axes[1, 1], color=['blue', 'red'], edgecolor='black')
axes[1, 1].set_title('Deserción por Becado', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Becado', fontsize=11)
axes[1, 1].set_ylabel('Cantidad', fontsize=11)
axes[1, 1].legend(title='Desertó')
axes[1, 1].grid(True, alpha=0.3, axis='y')
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig(OUTDIR / "distribuciones.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Guardado: distribuciones.png")

# Boxplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, col in enumerate(NUM_COLS):
    df.boxplot(column=col, by='Desertó', ax=axes[idx])
    axes[idx].set_title(f'{col} por Deserción', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Desertó', fontsize=11)
    axes[idx].set_ylabel(col, fontsize=11)
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('')
plt.tight_layout()
plt.savefig(OUTDIR / "boxplots.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Guardado: boxplots.png")

# Scatter plots
fig, ax = plt.subplots(figsize=(10, 6))
colors = {'No': 'blue', 'Sí': 'red'}
for desertion, group in df.groupby('Desertó'):
    ax.scatter(group['Promedio'], group['Materias_Perdidas'], 
               label=f'Desertó: {desertion}', 
               c=colors[desertion], 
               alpha=0.6, 
               edgecolors='black',
               s=50)
ax.set_xlabel('Promedio Académico', fontsize=12)
ax.set_ylabel('Materias Perdidas', fontsize=12)
ax.set_title('Promedio vs Materias Perdidas por Deserción', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTDIR / "scatter_promedio_materias.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Guardado: scatter_promedio_materias.png")

# ==========================================================
# 4️⃣ PROCESAMIENTO
# ==========================================================

print("\n=== 🔧 PROCESAMIENTO DE DATOS ===")

# Copiar dataset para procesamiento
df_proc = df.copy()

# Convertir variables categóricas a numéricas
df_proc['Becado'] = df_proc['Becado'].map({'Sí': 1, 'No': 0})
df_proc['Desertó'] = df_proc['Desertó'].map({'Sí': 1, 'No': 0})

# Verificar y manejar valores nulos (aunque no debería haber)
for c in NUM_COLS:
    df_proc[c] = pd.to_numeric(df_proc[c], errors="coerce")
    if df_proc[c].isna().sum() > 0:
        print(f"⚠️  Imputando {df_proc[c].isna().sum()} valores nulos en '{c}' con la mediana")
        df_proc[c] = df_proc[c].fillna(df_proc[c].median())

# Separar features y target
X = df_proc.drop(columns=[TARGET])
y = df_proc[TARGET]

print(f"\nShape de X: {X.shape}")
print(f"Shape de y: {y.shape}")
print(f"\nColumnas en X: {list(X.columns)}")

# ==========================================================
# 5️⃣ DIVISIÓN DEL DATASET (70% / 15% / 15%)
# ==========================================================

print("\n=== ✂️  DIVISIÓN DEL DATASET ===")

# Split inicial: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# Split de temp: 50% val, 50% test (15% y 15% del total)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print(f"\nTrain: {len(X_train)} muestras ({len(X_train)/len(X)*100:.1f}%)")
print(f"Val:   {len(X_val)} muestras ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test:  {len(X_test)} muestras ({len(X_test)/len(X)*100:.1f}%)")

print(f"\nDistribución en Train:")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True))

# ---------------------------
# Balanceo del conjunto de TRAIN
# ---------------------------
print("\n--- Balanceando conjunto de TRAIN ---")

train_df = pd.concat([X_train, y_train], axis=1)
class_0 = train_df[train_df[TARGET] == 0]
class_1 = train_df[train_df[TARGET] == 1]

print(f"\nAntes del balanceo:")
print(f"  Clase 0 (No Desertó): {len(class_0)}")
print(f"  Clase 1 (Desertó):    {len(class_1)}")

min_class = min(len(class_0), len(class_1))

# Under-sampling de la clase mayoritaria
class_0_bal = resample(class_0, replace=False, n_samples=min_class, random_state=42)
class_1_bal = resample(class_1, replace=False, n_samples=min_class, random_state=42)

train_balanced = pd.concat([class_0_bal, class_1_bal])

X_train = train_balanced.drop(columns=[TARGET])
y_train = train_balanced[TARGET]

print(f"\nDespués del balanceo:")
print(f"  Clase 0: {(y_train == 0).sum()}")
print(f"  Clase 1: {(y_train == 1).sum()}")
print(f"  Total Train balanceado: {len(X_train)}")

# ---------------------------
# Escalado (StandardScaler)
# ---------------------------
print("\n--- Escalando variables numéricas ---")

scaler = StandardScaler()
X_train_copy = X_train.copy()
X_val_copy = X_val.copy()
X_test_copy = X_test.copy()

X_train_copy[NUM_COLS] = scaler.fit_transform(X_train[NUM_COLS])
X_val_copy[NUM_COLS]   = scaler.transform(X_val[NUM_COLS])
X_test_copy[NUM_COLS]  = scaler.transform(X_test[NUM_COLS])

X_train = X_train_copy
X_val = X_val_copy
X_test = X_test_copy

print("✓ Escalado completado (StandardScaler)")

# Guardar estadísticas del scaler
scaler_stats = {
    col: {
        "mean": float(scaler.mean_[idx]),
        "std": float(scaler.scale_[idx])
    }
    for idx, col in enumerate(NUM_COLS)
}

with open(OUTDIR / "scaler_stats.json", "w", encoding="utf-8") as f:
    json.dump(scaler_stats, f, indent=2, ensure_ascii=False)

# ==========================================================
# 6️⃣ EXPORTACIÓN DE DATOS PROCESADOS
# ==========================================================

print("\n=== 💾 EXPORTANDO DATOS PROCESADOS ===")

# Exportar en CSV
X_train.to_csv(OUTDIR / "X_train.csv", index=False)
X_val.to_csv(OUTDIR / "X_val.csv", index=False)
X_test.to_csv(OUTDIR / "X_test.csv", index=False)

y_train.to_frame(name=TARGET).to_csv(OUTDIR / "y_train.csv", index=False)
y_val.to_frame(name=TARGET).to_csv(OUTDIR / "y_val.csv", index=False)
y_test.to_frame(name=TARGET).to_csv(OUTDIR / "y_test.csv", index=False)

print("✓ Guardados archivos .csv")

# Metadata del procesamiento
schema = {
    "split_ratio": "70 / 15 / 15",
    "train_samples": len(X_train),
    "val_samples": len(X_val),
    "test_samples": len(X_test),
    "train_balance": {
        "Clase_0_No_Desertó": int((y_train == 0).sum()),
        "Clase_1_Desertó": int((y_train == 1).sum())
    },
    "val_distribution": {
        "Clase_0_No_Desertó": int((y_val == 0).sum()),
        "Clase_1_Desertó": int((y_val == 1).sum())
    },
    "test_distribution": {
        "Clase_0_No_Desertó": int((y_test == 0).sum()),
        "Clase_1_Desertó": int((y_test == 1).sum())
    },
    "features": list(X_train.columns),
    "num_features": len(X_train.columns),
    "scaler": "StandardScaler",
    "balancing": "Under-sampling en Train"
}

with open(OUTDIR / "processed_schema.json", "w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2, ensure_ascii=False)

print("✓ Guardado: processed_schema.json")

# ==========================================================
# 7️⃣ RESUMEN FINAL
# ==========================================================

print("\n" + "="*70)
print("✅ PIPELINE COMPLETO EJECUTADO CORRECTAMENTE")
print("="*70)

print(f"\n📂 Todos los archivos guardados en: {OUTDIR}/")
print("\nArchivos generados:")
print("  📊 EDA y Estadísticas:")
print("     - definicion_problema.json")
print("     - descripcion_basica.csv")
print("     - tendencia_central_numericas.csv")
print("     - moda_categoricas.json")
print("     - proporciones_categoricas.json")
print("     - iqr_results.json")
print("     - percentiles.json")
print("     - correlation_stats.json")
print("     - pivot_promedio_beca_desercion.csv")
print("     - pivot_materias_beca_desercion.csv")
print("     - pivot_count_beca_desercion.csv")
print("\n  📈 Visualizaciones:")
print("     - heatmap_correlacion.png")
print("     - distribuciones.png")
print("     - boxplots.png")
print("     - scatter_promedio_materias.png")
print("\n  💾 Datos Procesados:")
print("     - X_train.csv, X_val.csv, X_test.csv")
print("     - y_train.csv, y_val.csv, y_test.csv")
print("     - processed_schema.json")
print("     - scaler_stats.json")

print("\n" + "="*70)
print("📚 SIGUIENTE PASO: Entrenamiento del modelo (XGBoost)")
print("="*70)
print("\n💡 Recomendaciones:")
print("   1. Analizar correlaciones en heatmap_correlacion.png")
print("   2. Verificar distribuciones por clase en distribuciones.png")
print("   3. Revisar pivot tables para insights")
print("   4. Proceder con entrenamiento usando X_train.csv y y_train.csv")
print("   5. Recordar: Métrica principal = RECALL (>80%)")
