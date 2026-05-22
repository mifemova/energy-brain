"""
================================================================================
MÓDULO F2 — Entrenamiento y Evaluación de Modelos (OE2)
================================================================================

Este módulo entrena y evalúa los tres algoritmos de ensamble comparados en el
piloto experimental: Random Forest, XGBoost y CatBoost. La evaluación se
realiza mediante validación cruzada estratificada de diez pliegues, garantizando
que cada pliegue preserve la proporción original de la clase de abandono.

Decisión metodológica crítica: SMOTE se aplica EXCLUSIVAMENTE sobre el pliegue
de entrenamiento dentro de cada iteración del cross-validation. Aplicar SMOTE
sobre el dataset completo antes de la división causaría filtración de
información sintética hacia el conjunto de prueba e inflaría artificialmente
las métricas de recall (Dong et al., 2025; Ovtšarenko, 2026). Este patrón se
implementa mediante imblearn.pipeline.Pipeline, que aplica los transformadores
únicamente en la fase fit y no en la fase predict.

Métricas reportadas para cada algoritmo:
    - AUC-ROC: capacidad discriminativa global del modelo.
    - Recall (sensibilidad) sobre la clase Withdrawn: porcentaje de
      estudiantes en riesgo correctamente identificados; métrica prioritaria
      desde el punto de vista pedagógico, pues un falso negativo equivale a
      no intervenir cuando se debería.
    - F1-score ponderado: balance entre precisión y recall ajustado por el
      tamaño de cada clase, robusto al desbalance estructural del dataset.

Nota de implementación (Windows + paralelismo):
    El clasificador CatBoost se configura con `allow_writing_files=False`
    para evitar una condición de carrera del sistema de archivos al ejecutar
    múltiples instancias paralelas dentro del GridSearchCV en entornos
    Windows. Esta configuración desactiva la escritura de logs internos del
    algoritmo en una carpeta compartida ('catboost_info'), sin afectar a los
    hiperparámetros entrenados ni a las métricas reportadas.
================================================================================
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import roc_auc_score, recall_score, f1_score

# Dependencias opcionales: se cargan solo si están instaladas
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

logger = logging.getLogger("F2_Modeling")
RANDOM_STATE: int = 42


# ============================================================================
# 1. CATÁLOGO DE MODELOS Y ESPACIO DE HIPERPARÁMETROS
# ============================================================================

@dataclass
class ModelSpec:
    """
    Especificación declarativa de un modelo a entrenar.

    Encapsula el estimador, su grilla de hiperparámetros y una descripción
    académica de por qué se incluye en la comparación. Mantener estas tres
    piezas juntas evita inconsistencias entre lo que el código entrena y lo
    que la memoria del TFM declara entrenar.
    """
    name: str
    estimator: Any
    param_grid: dict
    rationale: str


def get_model_catalog() -> dict[str, ModelSpec]:
    """
    Devuelve el catálogo de los tres modelos del piloto experimental.

    Las grillas de hiperparámetros se mantienen deliberadamente pequeñas
    (entre 8 y 18 combinaciones por modelo) para que el GridSearchCV anidado
    sea computacionalmente viable en un entorno académico. Cada combinación
    se evalúa con StratifiedKFold interno de 3 pliegues; combinado con los
    10 pliegues externos, el coste total se mantiene por debajo de las
    10 × 3 × 18 = 540 invocaciones de entrenamiento por modelo, asumibles
    en una máquina con 16 GB de RAM y 4 horas de cómputo.
    """
    catalog: dict[str, ModelSpec] = {}

    # ── Random Forest ──────────────────────────────────────────────────
    catalog["RandomForest"] = ModelSpec(
        name="RandomForest",
        estimator=RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight=None,  # SMOTE ya equilibra las clases en cada pliegue
        ),
        param_grid={
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [None, 12, 20],
            "clf__min_samples_split": [2, 5],
            "clf__max_features": ["sqrt"],
        },
        rationale=(
            "Algoritmo más utilizado en la predicción de abandono según "
            "Andrade-Girón et al. (2023). Sirve como modelo de referencia."
        ),
    )

    # ── XGBoost ─────────────────────────────────────────────────────────
    if XGBOOST_AVAILABLE:
        catalog["XGBoost"] = ModelSpec(
            name="XGBoost",
            estimator=XGBClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric="logloss",
                tree_method="hist",
                use_label_encoder=False,
                verbosity=0,
            ),
            param_grid={
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [4, 6, 8],
                "clf__learning_rate": [0.05, 0.1],
                "clf__subsample": [0.8, 1.0],
            },
            rationale=(
                "Algoritmo de mejor desempeño en Patel y Amin (2024) sobre el "
                "mismo conjunto OULAD (87 % de exactitud)."
            ),
        )

    # ── CatBoost ───────────────────────────────────────────────────────
    if CATBOOST_AVAILABLE:
        catalog["CatBoost"] = ModelSpec(
            name="CatBoost",
            estimator=CatBoostClassifier(
                random_state=RANDOM_STATE,
                verbose=False,
                thread_count=-1,
                auto_class_weights=None,
                # ──────────────────────────────────────────────────────────
                # Desactiva la escritura de logs internos del entrenador.
                # En Windows, múltiples instancias paralelas dentro del
                # GridSearchCV intentan crear simultáneamente la carpeta
                # 'catboost_info' en el directorio de trabajo, lo que produce
                # una condición de carrera del sistema de archivos y aborta
                # entre un 5 % y un 15 % de los fits internos, sesgando la
                # selección de hiperparámetros. Esta opción no afecta a los
                # hiperparámetros entrenados ni a las métricas reportadas;
                # solo suprime la telemetría interna del algoritmo.
                # ──────────────────────────────────────────────────────────
                allow_writing_files=False,
            ),
            param_grid={
                "clf__iterations": [300, 500],
                "clf__depth": [4, 6, 8],
                "clf__learning_rate": [0.05, 0.1],
                "clf__l2_leaf_reg": [3, 5],
            },
            rationale=(
                "Mejor AUC-ROC reportado por Vinces-Vinces y Flores-Sánchez "
                "(2025) en comparación de tres modelos de ensamble; maneja "
                "categóricas de forma nativa."
            ),
        )

    return catalog


# ============================================================================
# 2. RESULTADO DEL ENTRENAMIENTO DE UN MODELO
# ============================================================================

@dataclass
class ModelResult:
    """Empaqueta los resultados completos del entrenamiento de un algoritmo."""
    name: str
    best_estimator: Any                  # Modelo ya ajustado con mejores hyperparams
    best_params: dict                    # Combinación ganadora
    cv_metrics: dict                     # Media y desviación por pliegue
    per_fold_metrics: pd.DataFrame       # Métricas pliegue a pliegue
    training_time_seconds: float


# ============================================================================
# 3. ENTRENAMIENTO Y VALIDACIÓN CRUZADA ANIDADA
# ============================================================================

class CrossValidatedTrainer:
    """
    Entrena un modelo con validación cruzada estratificada anidada y SMOTE
    aplicado únicamente sobre el pliegue de entrenamiento.

    Arquitectura del bucle:
        1. StratifiedKFold externo (10 pliegues) divide los datos en train/test.
        2. Para cada pliegue externo, el ImbPipeline encadena:
             a) SMOTE sobre el train del pliegue (no toca el test).
             b) Estimador (RF, XGB o CatBoost) con sus hiperparámetros.
        3. GridSearchCV interno (3 pliegues estratificados) recorre la grilla
           y elige la mejor combinación sobre el train del pliegue externo.
        4. Las métricas se calculan sobre el test del pliegue externo, nunca
           tocado por SMOTE.

    Este diseño es el "gold standard" para evitar data leakage en problemas
    de clasificación desbalanceada con técnicas de oversampling.
    """

    def __init__(
        self,
        n_outer_folds: int = 10,
        n_inner_folds: int = 3,
        scoring_metric: str = "roc_auc",
    ):
        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds
        self.scoring_metric = scoring_metric

    def train(
        self,
        model_spec: ModelSpec,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> ModelResult:
        """Entrena un modelo siguiendo el protocolo de CV anidado."""
        logger.info("Iniciando entrenamiento de %s …", model_spec.name)
        t_start = time.time()

        # Pipeline imblearn-compatible: SMOTE + clasificador
        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
            ("clf", model_spec.estimator),
        ])

        outer_cv = StratifiedKFold(
            n_splits=self.n_outer_folds, shuffle=True, random_state=RANDOM_STATE,
        )
        inner_cv = StratifiedKFold(
            n_splits=self.n_inner_folds, shuffle=True, random_state=RANDOM_STATE,
        )

        # GridSearchCV interno: busca mejores hiperparámetros sobre cada train fold
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=model_spec.param_grid,
            scoring=self.scoring_metric,
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )

        # cross_validate ejecuta el outer CV y, dentro de cada pliegue,
        # GridSearchCV ajusta los hiperparámetros sobre el train fold.
        cv_results = cross_validate(
            estimator=grid,
            X=X, y=y,
            cv=outer_cv,
            scoring={
                "auc_roc": "roc_auc",
                "recall_withdrawn": "recall",
                "f1_weighted": "f1_weighted",
                "precision_withdrawn": "precision",
            },
            return_estimator=True,
            n_jobs=1,  # n_jobs=-1 ya se usa dentro del GridSearchCV
            verbose=0,
        )

        # Extraemos métricas por pliegue para análisis posterior
        per_fold = pd.DataFrame({
            "fold": range(1, self.n_outer_folds + 1),
            "auc_roc": cv_results["test_auc_roc"],
            "recall_withdrawn": cv_results["test_recall_withdrawn"],
            "f1_weighted": cv_results["test_f1_weighted"],
            "precision_withdrawn": cv_results["test_precision_withdrawn"],
        })

        cv_metrics = {
            "auc_roc_mean":          float(per_fold["auc_roc"].mean()),
            "auc_roc_std":           float(per_fold["auc_roc"].std()),
            "recall_withdrawn_mean": float(per_fold["recall_withdrawn"].mean()),
            "recall_withdrawn_std":  float(per_fold["recall_withdrawn"].std()),
            "f1_weighted_mean":      float(per_fold["f1_weighted"].mean()),
            "f1_weighted_std":       float(per_fold["f1_weighted"].std()),
            "precision_withdrawn_mean": float(per_fold["precision_withdrawn"].mean()),
        }

        # Refit final con la mejor combinación sobre TODO el dataset, para
        # usarlo después en el análisis SHAP (F3). Esta refitting es necesaria
        # porque cross_validate devuelve un estimador por pliegue, pero para
        # SHAP queremos un modelo único entrenado con todos los datos posibles.
        grid.fit(X, y)
        best_estimator = grid.best_estimator_
        best_params = grid.best_params_

        elapsed = time.time() - t_start
        logger.info(
            "%s entrenado en %.1f s — AUC-ROC = %.4f (±%.4f), Recall = %.4f",
            model_spec.name, elapsed,
            cv_metrics["auc_roc_mean"], cv_metrics["auc_roc_std"],
            cv_metrics["recall_withdrawn_mean"],
        )

        return ModelResult(
            name=model_spec.name,
            best_estimator=best_estimator,
            best_params=best_params,
            cv_metrics=cv_metrics,
            per_fold_metrics=per_fold,
            training_time_seconds=elapsed,
        )


# ============================================================================
# 4. ORQUESTADOR DE LA COMPARACIÓN DE LOS TRES MODELOS
# ============================================================================

class ModelComparator:
    """
    Orquesta el entrenamiento y la comparación de los tres modelos del piloto.

    Produce una tabla comparativa lista para incluirse como Tabla en el
    Capítulo 5 del TFM (Descripción de los resultados).
    """

    def __init__(
        self,
        n_outer_folds: int = 10,
        n_inner_folds: int = 3,
    ):
        self.trainer = CrossValidatedTrainer(
            n_outer_folds=n_outer_folds,
            n_inner_folds=n_inner_folds,
        )

    def run_all(self, X: pd.DataFrame, y: pd.Series) -> dict[str, ModelResult]:
        """Entrena los tres modelos y devuelve el diccionario de resultados."""
        catalog = get_model_catalog()
        results: dict[str, ModelResult] = {}

        for name, spec in catalog.items():
            logger.info("=" * 70)
            logger.info("MODELO %s: %s", name, spec.rationale)
            logger.info("=" * 70)
            results[name] = self.trainer.train(spec, X, y)

        return results

    @staticmethod
    def build_comparison_table(results: dict[str, ModelResult]) -> pd.DataFrame:
        """
        Construye la tabla comparativa de los tres modelos.

        Esta tabla puede insertarse directamente en el Capítulo 5 de la memoria.
        """
        rows = []
        for name, r in results.items():
            rows.append({
                "Modelo": name,
                "AUC-ROC (media)":   round(r.cv_metrics["auc_roc_mean"], 4),
                "AUC-ROC (desv.)":   round(r.cv_metrics["auc_roc_std"], 4),
                "Recall Withdrawn":  round(r.cv_metrics["recall_withdrawn_mean"], 4),
                "F1 ponderado":      round(r.cv_metrics["f1_weighted_mean"], 4),
                "Precisión Withdrawn": round(r.cv_metrics["precision_withdrawn_mean"], 4),
                "Tiempo (s)":        round(r.training_time_seconds, 1),
            })
        return pd.DataFrame(rows).sort_values("AUC-ROC (media)", ascending=False)


if __name__ == "__main__":
    print("Módulo F2 cargado correctamente.")
    print(f"XGBoost disponible: {XGBOOST_AVAILABLE}")
    print(f"CatBoost disponible: {CATBOOST_AVAILABLE}")
    catalog = get_model_catalog()
    print(f"Modelos a entrenar: {list(catalog.keys())}")
