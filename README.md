# Aplicación de la Inteligencia Artificial en la Innovación Educativa
## Piloto experimental — Predicción del abandono estudiantil en entornos virtuales

**Trabajo Fin de Máster · Universidad Internacional de La Rioja (UNIR)**
*Máster Universitario en Inteligencia Artificial — Tipo 1: Piloto Experimental*

Autores: Gloria Lida Alzate Suárez · Miguel Fernando Morales Valderrama
Director: Víctor Daniel Diaz Suarez

---

## 1. Propósito del repositorio

Este repositorio contiene el código fuente, los cuadernos reproducibles y el dashboard interactivo que materializan el piloto experimental descrito en el Capítulo 3 de la memoria del TFM. El piloto compara el desempeño predictivo de tres modelos de clasificación supervisada (Random Forest, XGBoost y CatBoost) sobre el *Open University Learning Analytics Dataset* (OULAD), interpreta las predicciones con el método SHAP, valida estadísticamente la consistencia de los predictores entre algoritmos y traduce los hallazgos en recomendaciones pedagógicas validadas con tutores expertos.

## 2. Arquitectura del proyecto

```
tfm_oulad/
├── src/
│   ├── f1_preprocessing.py            # OE1: Integración OULAD, features LMS y evaluativas
│   ├── f2_modeling.py                 # OE2: RF + XGBoost + CatBoost con CV anidado
│   ├── f3_shap_analysis.py            # OE3: TreeExplainer y visualizaciones SHAP
│   ├── f4_statistical_consistency.py  # OE4: Kruskal-Wallis + Mann-Whitney + Bonferroni
│   └── f5_pedagogy.py                 # OE5: Perfiles de riesgo + cuestionario Likert
├── notebooks/
│   └── 00_full_pipeline.ipynb         # Cuaderno orquestador reproducible
├── dashboard/
│   └── dashboard.py                   # Dashboard interactivo Streamlit
├── results/                           # Artefactos generados (CSV, JSON, PNG)
├── main_pipeline.py                   # Orquestador desde línea de comandos
├── requirements.txt                   # Versiones fijadas para reproducibilidad
└── README.md                          # Este archivo
```

## 3. Requisitos del entorno

- **Python 3.10 o 3.11** (probado en ambos)
- **Sistema operativo:** Linux, macOS o Windows
- **RAM mínima recomendada:** 16 GB (para el GridSearchCV anidado completo)
- **Almacenamiento:** ~500 MB libres para el OULAD descomprimido
- **Tiempo de cómputo:** entre 30 minutos y 4 horas según hardware y tamaño de las grillas de hiperparámetros

## 4. Instalación

Se recomienda trabajar en un entorno virtual aislado:

```bash
git clone <url-del-repositorio> tfm_oulad
cd tfm_oulad

python -m venv .venv
source .venv/bin/activate              # En Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Obtención del Open University Learning Analytics Dataset

El OULAD es de acceso abierto y puede descargarse de dos formas:

**Opción A — Fuente oficial:** https://analyse.kmi.open.ac.uk/open_dataset

**Opción B — Kaggle CLI (recomendado si tienes cuenta y credenciales configuradas):**

```bash
kaggle datasets download -d anlgrbz/student-demographics-online-education-dataoulad --unzip -p data/oulad
```

Una vez descargado, verifica que los siete archivos CSV estén en la carpeta `data/oulad/`:

```
data/oulad/
├── assessments.csv
├── courses.csv
├── studentAssessment.csv
├── studentInfo.csv
├── studentRegistration.csv
├── studentVle.csv
└── vle.csv
```

## 6. Ejecución del pipeline

### 6.1 Desde la línea de comandos

```bash
python main_pipeline.py \
    --data-path ./data/oulad \
    --output-dir ./results \
    --n-outer-folds 10 \
    --n-inner-folds 3 \
    --shap-sample-size 3000
```

### 6.2 Desde el cuaderno reproducible

```bash
jupyter lab notebooks/00_full_pipeline.ipynb
```

Ejecuta las celdas secuencialmente. Cada fase guarda sus artefactos en `results/` y registra el avance en el log.

## 7. Dashboard interactivo

Una vez completada la ejecución del pipeline, lanza el dashboard:

```bash
streamlit run dashboard/dashboard.py -- --results-dir ./results
```

Se abrirá una ventana del navegador con cinco pestañas:

| Pestaña | Contenido |
|---|---|
| **Resumen** | Balance de clases, distribución de riesgo del mejor modelo |
| **Modelos** | Tabla comparativa, métricas por pliegue de la CV |
| **SHAP** | Importancia global y visualizaciones por modelo |
| **Consistencia** | Resultados Kruskal-Wallis y predictores robustos |
| **Pedagogía** | Catálogo de recomendaciones por perfil de riesgo |

## 8. Decisiones metodológicas relevantes

**Por qué SMOTE solo dentro del pliegue de entrenamiento.** Aplicar SMOTE sobre el dataset completo antes de la división causa filtración de información sintética hacia el conjunto de prueba e infla artificialmente las métricas de Recall. La librería `imblearn.pipeline.Pipeline` aplica SMOTE únicamente en la fase `fit` del pipeline, garantizando que los pliegues de evaluación nunca sean tocados por las muestras sintéticas.

**Por qué Kruskal-Wallis y no ANOVA.** Las distribuciones de valores SHAP por variable son fuertemente sesgadas y multimodales, lo que invalida los supuestos paramétricos de normalidad y homocedasticidad. Kruskal-Wallis compara medianas mediante rangos, robustez que conviene al objetivo de identificar predictores estables.

**Por qué la corrección de Bonferroni.** Realizar 60+ contrastes (20 variables × 3 algoritmos × pares) sin corrección eleva la probabilidad de detectar diferencias espurias a niveles inaceptables. Bonferroni divide el alpha global por el número total de pruebas, controlando la tasa de error familiar.

**Por qué TreeExplainer.** Los modelos comparados son todos basados en árboles. `shap.TreeExplainer` calcula valores SHAP exactos en tiempo polinomial aprovechando la estructura de los árboles, frente al tiempo exponencial del `KernelExplainer` agnóstico al modelo.

## 9. Artefactos producidos por el pipeline

Tras la ejecución completa, el directorio `results/` contiene:

- `preprocessed_features.parquet`: matriz X procesada
- `preprocessed_target.parquet`: vector y
- `class_balance.json`: distribución de clases
- `comparison_table.csv`: tabla comparativa de los tres modelos
- `cv_per_fold_<modelo>.csv`: métricas por pliegue (uno por modelo)
- `model_<modelo>.joblib`: modelos serializados con joblib
- `shap_summary_<modelo>.png`: summary plots SHAP
- `shap_dependence_<modelo>_top1.png` y `top2.png`: dependence plots
- `shap_waterfall_<modelo>_<nivel>.png`: waterfall plots (alto/medio/bajo)
- `shap_global_importance_<modelo>.csv`: importancia tabular
- `consistency_kruskal.csv`: resultados Kruskal-Wallis
- `consistency_pairwise.csv`: Mann-Whitney U corregido
- `robust_predictors.json`: clasificación final de predictores
- `risk_recommendations.json`: catálogo pedagógico completo
- `likert_questionnaire.csv`: cuestionario para validación con tutores
- `dashboard_data.json`: insumo único del dashboard

## 10. Cómo citar este trabajo

> Alzate Suárez, G. L., y Morales Valderrama, M. F. (2026). *Aplicación de la Inteligencia Artificial en la Innovación Educativa: Modelos predictivos y de clasificación para la mejora del aprendizaje* [Trabajo Fin de Máster, Universidad Internacional de La Rioja]. Repositorio institucional UNIR.

## 11. Licencia

Este código se distribuye bajo licencia académica abierta para fines de investigación y enseñanza, conforme a las directrices del Máster Universitario en Inteligencia Artificial de la UNIR. El dataset OULAD se rige por los términos de uso de la Open University del Reino Unido.

---

*Última actualización: mayo de 2026.*
