# 📊 Proyecto: Predicción de Deserción Estudiantil

## 🎯 Objetivo

Este proyecto implementa un **pipeline completo de ciencia de datos** para predecir la deserción estudiantil en la Universidad EAFIT, identificando tempranamente estudiantes en riesgo para aplicar intervenciones oportunas.

El modelo busca predecir:

> **¿Un estudiante desertará de la universidad? (0 = No, 1 = Sí)**

Se trata de un problema de:

* ✅ Clasificación binaria
* ✅ Aprendizaje supervisado
* ✅ Dataset estructurado
* ✅ Datos balanceados en entrenamiento

---

## 👥 Información del Proyecto

**Universidad:** EAFIT  
**Curso:** Fundamentos de Aprendizaje Automático  
**Tipo de Problema:** Clasificación binaria supervisada  
**Algoritmo Propuesto:** XGBoost (Gradient Boosting)  
**Métrica Principal:** **Recall (>80%)**  

### ¿Por qué Recall?

En este problema, un **Falso Negativo** (no detectar a un estudiante que va a desertar) es mucho más costoso que un **Falso Positivo** (falsa alarma). 

- **FN:** Perder un estudiante = ~$6.000 USD en matrícula perdida
- **FP:** Ofrecer ayuda innecesaria = ~$100 USD

**Ratio de costo:** 60:1

Por eso priorizamos **detectar todos los casos de deserción** aunque tengamos algunas falsas alarmas.

---

# 1️⃣ Definición del Problema

Se define formalmente el problema en un archivo:

```
data_output_desercion/definicion_problema.json
```

Contiene:

* Objetivo del proyecto
* Impacto esperado
* Tipo de problema (clasificación binaria)
* Variables utilizadas
* Algoritmo propuesto (XGBoost)
* Métrica principal (Recall)
* Justificación de la métrica

### Variables utilizadas

#### Variables Numéricas

* **Promedio:** Promedio académico acumulado (escala 0.0 - 5.0)
* **Materias_Perdidas:** Número de materias reprobadas (0 - 6)

#### Variables Categóricas (Binarias)

* **Becado:** Si el estudiante tiene beca (Sí/No)

#### Variable Objetivo (Target)

* **Desertó:** Si el estudiante desertó (Sí/No)

---

# 2️⃣ Generación y Recolección de Datos

## Dataset Sintético Realista

Se generó un dataset de **500 estudiantes** con las siguientes características:

```python
dataset_desercion_estudiantes.csv
```

### Lógica de Generación

Los datos fueron generados con **correlaciones realistas**:

1. **Promedio bajo → Mayor riesgo de deserción**
   - Promedio < 2.5: Alto riesgo
   - Promedio 3.0-3.5: Riesgo moderado
   - Promedio > 4.0: Bajo riesgo

2. **Materias perdidas → Mayor riesgo**
   - Cada materia perdida aumenta el riesgo en ~8%

3. **Beca → Efecto protector**
   - Tener beca reduce el riesgo de deserción en 60%

### Estadísticas del Dataset

| Métrica | Valor |
|---------|-------|
| Total de estudiantes | 500 |
| Tasa de deserción | 32.8% (164 estudiantes) |
| Estudiantes becados | 33.0% (165 estudiantes) |
| Rango de IDs | 1000 - 1499 |
| Promedio general | 3.49 |
| Materias perdidas (promedio) | 1.38 |

### Diferencias entre Desertores y No Desertores

| Variable | Desertores | No Desertores | Diferencia |
|----------|-----------|---------------|------------|
| Promedio | 2.99 | 3.73 | **+0.74** |
| Materias perdidas | 2.33 | 0.91 | **+1.42** |
| % Becados | 15.2% | 41.7% | **-26.5 p.p.** |

Esto permite verificar:

* Calidad de los datos ✓
* Balance de clases (67.2% / 32.8%)
* Correlaciones realistas ✓
* Patrones detectables ✓

---

# 3️⃣ Análisis Exploratorio de Datos (EDA)

Se realiza un análisis estadístico completo con **script independiente**:

```bash
python eda_desercion_estudiantes.py
```

---

## Tendencia Central

Para variables numéricas:

* **Media**
* **Mediana** 
* **Moda**

Archivos generados:

```
eda_output/01_tendencia_central_numericas.csv
```

Para categóricas:

```
eda_output/01_moda_categoricas.json
eda_output/01_proporciones_categoricas.json
```

**Ejemplo de resultados:**

| Variable | Media | Mediana | Moda |
|----------|-------|---------|------|
| Promedio | 3.49 | 3.52 | 3.40 |
| Materias_Perdidas | 1.38 | 1.00 | 0.00 |

---

## Cuartiles e IQR

Se calcula:

* **Q1** (Percentil 25)
* **Q2** (Mediana)
* **Q3** (Percentil 75)
* **IQR** (Rango Intercuartílico) = Q3 - Q1
* **Límites de outliers:** [Q1 - 1.5×IQR, Q3 + 1.5×IQR]

Archivo:

```
eda_output/02_iqr_results.json
```

**Utilidad:** Identifica valores atípicos y comprende la dispersión de los datos.

---

## Percentiles

Se calculan:

* **P10** (10% de los datos están por debajo)
* **P25** (Q1)
* **P50** (Mediana)
* **P75** (Q3)
* **P90** (90% de los datos están por debajo)

Archivo:

```
eda_output/03_percentiles.json
```

**Utilidad:** Entender la distribución completa de los datos.

---

## Correlaciones

Se genera:

* **Matriz de correlación completa**
* **Heatmap visual** (PNG)
* **Correlación Pearson** (lineal)
* **Correlación Spearman** (monotónica)

Archivos generados:

```
eda_output/04_heatmap_correlacion.png
eda_output/04_correlation_stats.json
```

### Correlaciones encontradas

| Variable | Pearson | Spearman | Interpretación |
|----------|---------|----------|----------------|
| Promedio | -0.51 | -0.53 | **Negativa fuerte:** Promedio bajo → Mayor deserción |
| Materias_Perdidas | +0.67 | +0.69 | **Positiva fuerte:** Más materias perdidas → Mayor deserción |
| Becado | -0.29 | -0.29 | **Negativa moderada:** Tener beca → Menor deserción |

Esto permite entender:

* ✅ Qué variables impactan más el target
* ✅ Relaciones lineales vs monotónicas
* ✅ Multicolinealidad entre features

---

## Tablas Pivote (Pivot Tables)

Se analizan agregaciones cruzadas:

### Promedio por Beca y Deserción

```
eda_output/05_pivot_promedio_beca_desercion.csv
```

| Becado | No Desertó | Desertó |
|--------|-----------|---------|
| No | 3.64 | 3.04 |
| Sí | 3.99 | 2.77 |

**Insight:** Los becados tienen mejor promedio, pero si desertan, su promedio sigue siendo bajo.

### Materias Perdidas por Beca y Deserción

```
eda_output/05_pivot_materias_beca_desercion.csv
```

---

## Visualizaciones Estáticas (PNG)

Se generan 6 gráficos profesionales:

### 1. Heatmap de Correlación

```
eda_output/04_heatmap_correlacion.png
```

Muestra la matriz de correlación con colores (rojo = positiva, azul = negativa).

### 2. Histogramas de Distribución

```
eda_output/06_histogramas_distribuciones.png
```

4 subplots:
- Promedio por deserción
- Materias perdidas por deserción
- Conteo por deserción
- Deserción por beca (barras)

### 3. Boxplots por Deserción

```
eda_output/06_boxplots_por_desercion.png
```

Muestra la distribución de Promedio y Materias Perdidas separadas por clase.

### 4. Scatter Plot: Promedio vs Materias

```
eda_output/06_scatter_promedio_materias.png
```

Visualización 2D mostrando la separación entre desertores (rojo) y no desertores (azul).

### 5. Barras de Proporciones

```
eda_output/06_barras_proporciones.png
```

### 6. Gráfico Stacked

```
eda_output/06_stacked_desercion_beca.png
```

Barras apiladas mostrando deserción por beca.

---

## Resumen Estadístico por Clase

Se generan estadísticas descriptivas separadas:

```
eda_output/07_resumen_estadistico_por_clase.csv
eda_output/07_comparacion_medias_por_clase.csv
```

**Ejemplo:**

| Estadística | Desertores | No Desertores |
|-------------|-----------|---------------|
| Promedio (media) | 2.99 | 3.73 |
| Materias (media) | 2.33 | 0.91 |
| Becados (%) | 15.2% | 41.7% |

---

## Identificación de Outliers

Se detectan valores atípicos usando el método **IQR**:

```
eda_output/08_outliers_info.json
```

**Criterio:** Un valor es outlier si:
- Está por debajo de Q1 - 1.5×IQR
- Está por encima de Q3 + 1.5×IQR

---

# 4️⃣ Procesamiento de Datos

Script principal:

```bash
python pipeline_desercion_estudiantes.py
```

Se realiza:

### Limpieza

* **Conversión segura a numérico**
* **Imputación con mediana** (variables numéricas, si hay nulos)
* **Manejo de valores faltantes** en categóricas

### Encoding

Se convierte:

```python
Becado: Sí → 1, No → 0
Desertó: Sí → 1, No → 0
```

**Resultado:** Todas las variables son numéricas para el modelo.

---

# 5️⃣ División del Dataset

Se aplica split estratificado:

```
70% Train (350 muestras)
15% Validation (75 muestras)
15% Test (75 muestras)
```

Con:

```python
stratify=y  # Mantiene la proporción de clases
random_state=42  # Reproducibilidad
```

Esto garantiza que la proporción de desertores/no desertores se mantenga en todos los splits.

### Distribución de Clases

| Split | Clase 0 (No Desertó) | Clase 1 (Desertó) |
|-------|---------------------|-------------------|
| Train (antes balanceo) | 235 | 115 |
| Train (después balanceo) | 115 | 115 |
| Validation | 50 | 25 |
| Test | 51 | 24 |

---

# ⚖️ Balanceo de Clases (Solo Train)

Se utiliza **Under-sampling**:

```python
resample(replace=False, n_samples=min_class)
```

* Se reduce la clase mayoritaria al tamaño de la minoritaria
* Se evita que el modelo se sesgue hacia "No Desertó"
* Se mantiene la información más valiosa

### ¿Por qué balancear?

Sin balanceo, el modelo podría:
- Predecir siempre "No Desertó" 
- Obtener 67% de accuracy
- Pero tener 0% de Recall (¡no detecta a ningún desertor!)

**Importante:**
El balanceo **solo se aplica en entrenamiento**, nunca en validación o test (para evaluar en condiciones reales).

---

# 📏 Escalado

Se usa **StandardScaler**:

```python
scaler = StandardScaler()
X_train[NUM_COLS] = scaler.fit_transform(X_train[NUM_COLS])
X_val[NUM_COLS] = scaler.transform(X_val[NUM_COLS])
X_test[NUM_COLS] = scaler.transform(X_test[NUM_COLS])
```

### ¿Qué hace StandardScaler?

Transforma cada variable a:
- **Media = 0**
- **Desviación estándar = 1**

### ¿Por qué escalar?

- El **Promedio** está en escala 0-5
- Las **Materias Perdidas** están en escala 0-6
- Sin escalar, el modelo podría dar más importancia a las materias por tener valores más grandes

### Regla de Oro

- **fit_transform** → Solo en train (aprende media y desviación)
- **transform** → En val y test (usa la media y desviación de train)

Esto evita **data leakage** (filtración de información del futuro).

---

# 6️⃣ Exportación Final

Se exportan los datos procesados en dos formatos:

### Formato Parquet (eficiente)

```
data_output_desercion/X_train.parquet
data_output_desercion/X_val.parquet
data_output_desercion/X_test.parquet
data_output_desercion/y_train.parquet
data_output_desercion/y_val.parquet
data_output_desercion/y_test.parquet
```

### Formato CSV (legible)

```
data_output_desercion/X_train.csv
data_output_desercion/X_val.csv
data_output_desercion/X_test.csv
data_output_desercion/y_train.csv
data_output_desercion/y_val.csv
data_output_desercion/y_test.csv
```

### Metadatos

```
data_output_desercion/processed_schema.json
data_output_desercion/scaler_stats.json
```

Contiene:

* Proporción de split (70/15/15)
* Balance final de clases
* Número de muestras por conjunto
* Features utilizadas
* Estadísticas del scaler (media y std)

---

# 📂 Estructura de Carpetas

```
proyecto-desercion-estudiantil/
│
├── 📄 README.md (este archivo)
│
├── 📊 Datos
│   ├── dataset_desercion_estudiantes.csv (dataset original)
│   └── data_output_desercion/ (datos procesados)
│       ├── definicion_problema.json
│       ├── descripcion_basica.csv
│       ├── scaler_stats.json
│       ├── processed_schema.json
│       ├── X_train.csv / X_train.parquet
│       ├── X_val.csv / X_val.parquet
│       ├── X_test.csv / X_test.parquet
│       ├── y_train.csv / y_train.parquet
│       ├── y_val.csv / y_val.parquet
│       └── y_test.csv / y_test.parquet
│
├── 📈 EDA (Análisis Exploratorio)
│   └── eda_output/
│       ├── 01_tendencia_central_numericas.csv
│       ├── 01_moda_categoricas.json
│       ├── 01_proporciones_categoricas.json
│       ├── 02_iqr_results.json
│       ├── 03_percentiles.json
│       ├── 04_heatmap_correlacion.png ⭐
│       ├── 04_correlation_stats.json
│       ├── 05_pivot_promedio_beca_desercion.csv
│       ├── 05_pivot_materias_beca_desercion.csv
│       ├── 05_pivot_count_beca_desercion.csv
│       ├── 06_histogramas_distribuciones.png ⭐
│       ├── 06_boxplots_por_desercion.png ⭐
│       ├── 06_scatter_promedio_materias.png ⭐
│       ├── 06_barras_proporciones.png
│       ├── 06_stacked_desercion_beca.png
│       ├── 07_resumen_estadistico_por_clase.csv
│       ├── 07_comparacion_medias_por_clase.csv
│       └── 08_outliers_info.json
│
└── 🐍 Scripts Python
    ├── pipeline_desercion_estudiantes.py (pipeline completo)
    └── eda_desercion_estudiantes.py (solo EDA)
```

---

# 🚀 Cómo Usar Este Proyecto

## Requisitos

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Ejecución

### Paso 1: Generar EDA (Análisis Exploratorio)

```bash
python eda_desercion_estudiantes.py
```

**Salida:** Carpeta `eda_output/` con estadísticas y visualizaciones

### Paso 2: Ejecutar Pipeline Completo

```bash
python pipeline_desercion_estudiantes.py
```

**Salida:** Carpeta `data_output_desercion/` con datos procesados listos para entrenar

### Paso 3: Entrenar Modelo (próximo paso)

```bash
# Próximamente
python train_model.py
```

---

# 📊 Resultados Esperados

## Insights del EDA

### 1. Promedio Académico

- **Desertores:** 2.99 (promedio bajo)
- **No Desertores:** 3.73 (promedio alto)
- **Diferencia:** 0.74 puntos
- **Correlación:** -0.51 (negativa fuerte)

### 2. Materias Perdidas

- **Desertores:** 2.33 materias
- **No Desertores:** 0.91 materias
- **Diferencia:** 1.42 materias
- **Correlación:** +0.67 (positiva fuerte)

### 3. Efecto de la Beca

- **Desertores becados:** 15.2%
- **No desertores becados:** 41.7%
- **Efecto protector:** -26.5 p.p.
- **Correlación:** -0.29 (negativa moderada)

### 💡 Conclusión del EDA

> Los desertores tienen **promedios más bajos**, pierden **más materias** y tienen **menor probabilidad de tener beca**. Estos patrones claros sugieren que un modelo de Machine Learning puede predecir la deserción exitosamente.

---

# 🎯 Buenas Prácticas Implementadas

| ✅ Práctica | Descripción |
|------------|-------------|
| **Separación clara de fases** | EDA, procesamiento y entrenamiento en scripts separados |
| **Sin data leakage** | Scaler fit solo en train, transform en val/test |
| **Balanceo correcto** | Solo en train, nunca en val/test |
| **Estratificación** | Split mantiene proporción de clases |
| **Escalado apropiado** | StandardScaler para variables numéricas |
| **Reproducibilidad** | random_state=42 en todos los splits |
| **Documentación completa** | EDA con estadísticas y visualizaciones |
| **Exportación eficiente** | CSV (legible) + Parquet (eficiente) |
| **Trazabilidad** | Metadatos en JSON |

---

# 📚 Próximos Pasos

## 7️⃣ Entrenamiento del Modelo

- [ ] Entrenar XGBoost con los datos balanceados
- [ ] Optimizar hiperparámetros con Grid Search
- [ ] Validar con conjunto de validación
- [ ] Ajustar threshold de decisión (0.3-0.4 en lugar de 0.5)

## 8️⃣ Evaluación

- [ ] Calcular métricas: Recall, Precision, F1-Score, AUC-ROC
- [ ] Generar matriz de confusión
- [ ] Analizar curva ROC
- [ ] Validar que Recall > 80%
- [ ] Interpretar feature importance

## 9️⃣ Despliegue (Opcional)

- [ ] Serializar modelo (pickle/joblib)
- [ ] Crear API REST (Flask/FastAPI)
- [ ] Dashboard interactivo (Streamlit)

## 🔟 Monitoreo (Opcional)

- [ ] Detectar data drift
- [ ] Detectar concept drift
- [ ] Reentrenamiento automático

---

# 🤝 Contribuciones

Este proyecto fue desarrollado como parte del curso **Fundamentos de Aprendizaje Automático** en la **Universidad EAFIT**.

---

# 📝 Licencia

Este proyecto es de uso académico.

---

# 📧 Contacto

Para preguntas o sugerencias sobre este proyecto, contactar a través de la plataforma académica de EAFIT.

---

## 🎓 Aprendizajes Clave

1. **Importancia del EDA:** Un buen análisis exploratorio es fundamental para entender los datos antes de modelar
2. **Balance de clases:** En problemas desbalanceados, el modelo puede sesgar hacia la clase mayoritaria
3. **Métricas apropiadas:** Accuracy no siempre es la mejor métrica (en este caso, Recall es más importante)
4. **Escalado:** Variables en diferentes escalas pueden sesgar el modelo
5. **Data leakage:** Información del test no debe filtrarse al entrenamiento
6. **Reproducibilidad:** random_state permite replicar resultados

---

**¡Gracias por revisar este proyecto! 🚀**
