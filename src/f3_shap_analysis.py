"""
================================================================================
MÓDULO F3 — Análisis de Explicabilidad SHAP (OE3)
================================================================================

Este módulo implementa la fase de explicabilidad del piloto experimental
mediante SHAP (SHapley Additive exPlanations), Lundberg y Lee (2017). El
método cuantifica la contribución marginal de cada variable a cada predicción
individual del modelo, transformando la opacidad inherente a los métodos de
ensamble en información interpretable para los profesionales de la educación.

Tres tipos de visualización se generan para cada uno de los tres modelos:

    1. SUMMARY PLOTS (resumen global). Ordenan las variables por su
       importancia SHAP media en valor absoluto, permitiendo identificar
       qué señales del LMS dominan globalmente la predicción de abandono.

    2. DEPENDENCE PLOTS (dependencia). Muestran la relación entre el valor
       de una variable y su contribución SHAP, revelando si los efectos son
       lineales, no-lineales o presentan interacciones con otras variables.

    3. WATERFALL PLOTS (cascada individual). Descomponen una predicción
       individual estudiante a estudiante, mostrando paso a paso cómo cada
       variable empuja la probabilidad de abandono hacia arriba o hacia
       abajo desde el valor base. Es la visualización pedagógicamente más
       potente porque genera el diagnóstico personalizado que los tutores
       pueden consultar para cada caso real.

El uso de TreeExplainer (en lugar de KernelExplainer) es obligado para
modelos basados en árboles: aprovecha la estructura de los árboles para
calcular valores SHAP exactos en tiempo polinomial, frente al tiempo
exponencial del método agnóstico al modelo.
================================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Backend no interactivo para entornos sin display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger("F3_SHAP")


# ============================================================================
# 1. ESTRUCTURAS DE DATOS
# ============================================================================

@dataclass
class SHAPArtifacts:
    """
    Empaqueta los productos del análisis SHAP de un único modelo.

    Conservamos tanto los valores SHAP brutos (para análisis estadístico en
    la fase F4) como las rutas a las figuras generadas (para incluirlas en
    el documento del TFM y en el dashboard interactivo).
    """
    model_name: str
    shap_values: np.ndarray           # Forma: (n_muestras, n_variables)
    feature_names: list[str]
    global_importance: pd.Series      # Importancia media absoluta por variable
    figure_paths: dict[str, str]      # nombre_figura → ruta_PNG


# ============================================================================
# 2. EXTRACCIÓN DEL ESTIMADOR FINAL DEL PIPELINE
# ============================================================================

def _extract_tree_model(pipeline_or_estimator: Any) -> Any:
    """
    Extrae el clasificador de árboles desde un pipeline imblearn-sklearn.

    Cuando se entrena un modelo en F2, el objeto devuelto es un ImbPipeline
    con dos pasos: 'smote' y 'clf'. TreeExplainer requiere el clasificador
    desnudo, no el pipeline. Esta función abstrae esa extracción y permite
    también recibir directamente un estimador, lo que facilita las pruebas
    unitarias del módulo.
    """
    if hasattr(pipeline_or_estimator, "named_steps"):
        return pipeline_or_estimator.named_steps["clf"]
    return pipeline_or_estimator


# ============================================================================
# 3. ANÁLISIS SHAP PARA UN MODELO
# ============================================================================

class SHAPAnalyzer:
    """
    Calcula valores SHAP y genera las visualizaciones requeridas por OE3.

    El parámetro sample_size permite trabajar con una muestra estratificada
    cuando el dataset completo es demasiado grande para el cálculo exacto.
    El OULAD íntegro contiene ~32.000 estudiantes; con TreeExplainer el
    cálculo sobre la matriz completa toma del orden de minutos para RF y
    XGBoost, y decenas de minutos para CatBoost. Para iteración rápida del
    experimento se recomienda sample_size = 3000, manteniendo la integridad
    estadística mediante muestreo estratificado por la variable objetivo.
    """

    def __init__(
        self,
        output_dir: str | Path,
        sample_size: int | None = None,
        random_state: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_size = sample_size
        self.random_state = random_state

    def _stratified_sample(
        self,
        X: pd.DataFrame, y: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Devuelve una muestra estratificada por la variable objetivo
        o el dataset completo si sample_size es None o supera len(X).

        Implementación basada en sklearn.model_selection.train_test_split
        con argumento stratify=y, lo que garantiza la conservación de la
        proporción de clases en la submuestra y evita problemas de
        indexado que pueden producirse al utilizar groupby.apply en
        versiones recientes de pandas.
        """
        if self.sample_size is None or self.sample_size >= len(X):
            return X.copy(), y.copy()
        from sklearn.model_selection import train_test_split
        _, X_sample, _, y_sample = train_test_split(
            X, y,
            test_size=self.sample_size,
            stratify=y,
            random_state=self.random_state,
        )
        return X_sample.reset_index(drop=True), y_sample.reset_index(drop=True)

    def analyze(
        self,
        model_name: str,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> SHAPArtifacts:
        """Ejecuta el análisis SHAP completo para un modelo."""
        logger.info("Analizando SHAP de %s …", model_name)

        # Muestreamos para acotar el coste computacional
        X_sample, y_sample = self._stratified_sample(X, y)

        # Extraemos el estimador del pipeline y construimos el explainer
        clf = _extract_tree_model(model)
        explainer = shap.TreeExplainer(clf)

        # Cálculo de valores SHAP. Para clasificación binaria, algunos
        # explicadores devuelven un array 3D (n_muestras, n_variables, n_clases)
        # y otros devuelven una lista por clase. Normalizamos a 2D
        # quedándonos siempre con los valores de la clase positiva (Withdrawn).
        raw_values = explainer.shap_values(X_sample)
        if isinstance(raw_values, list):
            shap_values = raw_values[1]  # Clase positiva
        elif raw_values.ndim == 3:
            shap_values = raw_values[:, :, 1]
        else:
            shap_values = raw_values

        feature_names = list(X_sample.columns)

        # Importancia global = media de |SHAP| por variable
        global_importance = pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=feature_names,
        ).sort_values(ascending=False)

        # Generamos las tres familias de visualizaciones
        figure_paths = {
            "summary": self._plot_summary(
                model_name, shap_values, X_sample, feature_names),
            "dependence_top1": self._plot_dependence(
                model_name, shap_values, X_sample,
                global_importance.index[0], suffix="top1"),
            "dependence_top2": self._plot_dependence(
                model_name, shap_values, X_sample,
                global_importance.index[1], suffix="top2"),
            "waterfall_high":  self._plot_waterfall(
                model_name, explainer, X_sample, shap_values, y_sample,
                risk_level="high"),
            "waterfall_medium": self._plot_waterfall(
                model_name, explainer, X_sample, shap_values, y_sample,
                risk_level="medium"),
            "waterfall_low":   self._plot_waterfall(
                model_name, explainer, X_sample, shap_values, y_sample,
                risk_level="low"),
        }

        return SHAPArtifacts(
            model_name=model_name,
            shap_values=shap_values,
            feature_names=feature_names,
            global_importance=global_importance,
            figure_paths=figure_paths,
        )

    # ── 3.1 SUMMARY PLOT ────────────────────────────────────────────────
    def _plot_summary(
        self,
        model_name: str,
        shap_values: np.ndarray,
        X_sample: pd.DataFrame,
        feature_names: list[str],
    ) -> str:
        """Genera el summary plot de importancia global."""
        fig = plt.figure(figsize=(9, 7))
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=feature_names,
            max_display=15,
            show=False,
        )
        plt.title(f"Importancia SHAP global — {model_name}", fontsize=12)
        out_path = self.output_dir / f"shap_summary_{model_name}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)

    # ── 3.2 DEPENDENCE PLOT ─────────────────────────────────────────────
    def _plot_dependence(
        self,
        model_name: str,
        shap_values: np.ndarray,
        X_sample: pd.DataFrame,
        feature: str,
        suffix: str,
    ) -> str:
        """Genera el dependence plot para una variable concreta."""
        fig = plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            feature, shap_values, X_sample,
            interaction_index="auto", show=False,
        )
        plt.title(f"Dependencia SHAP — {model_name} :: {feature}", fontsize=11)
        out_path = self.output_dir / f"shap_dependence_{model_name}_{suffix}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)

    # ── 3.3 WATERFALL PLOT ──────────────────────────────────────────────
    def _plot_waterfall(
        self,
        model_name: str,
        explainer: Any,
        X_sample: pd.DataFrame,
        shap_values: np.ndarray,
        y_sample: pd.Series,
        risk_level: str,
    ) -> str:
        """
        Selecciona un estudiante representativo del nivel de riesgo solicitado
        y genera su waterfall plot individual.

        Los niveles de riesgo se definen sobre la suma SHAP por instancia, que
        proxia el log-odds de la predicción del modelo:
            - high  : percentil 95 (mayor riesgo predicho)
            - medium: percentil 50 (riesgo intermedio)
            - low   : percentil 5  (menor riesgo predicho)
        """
        instance_scores = shap_values.sum(axis=1)
        if risk_level == "high":
            idx = int(np.argsort(instance_scores)[-int(0.05 * len(instance_scores))])
        elif risk_level == "low":
            idx = int(np.argsort(instance_scores)[int(0.05 * len(instance_scores))])
        else:  # medium
            idx = int(np.argsort(np.abs(
                instance_scores - np.median(instance_scores))).tolist()[0])

        # Extraemos el expected_value del explainer (puede ser escalar o array)
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1] if len(np.atleast_1d(expected_value)) > 1 \
                             else float(np.atleast_1d(expected_value)[0])

        explanation = shap.Explanation(
            values=shap_values[idx],
            base_values=expected_value,
            data=X_sample.iloc[idx].values,
            feature_names=list(X_sample.columns),
        )

        fig = plt.figure(figsize=(9, 7))
        shap.plots.waterfall(explanation, max_display=12, show=False)
        plt.title(
            f"Diagnóstico SHAP individual — {model_name} :: "
            f"Riesgo {risk_level} (estudiante #{idx})",
            fontsize=11,
        )
        out_path = self.output_dir / f"shap_waterfall_{model_name}_{risk_level}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)


# ============================================================================
# 4. ORQUESTADOR
# ============================================================================

def run_shap_analysis_all_models(
    trained_models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str | Path,
    sample_size: int | None = 3000,
) -> dict[str, SHAPArtifacts]:
    """Aplica el análisis SHAP a los tres modelos entrenados en F2."""
    analyzer = SHAPAnalyzer(output_dir=output_dir, sample_size=sample_size)
    results: dict[str, SHAPArtifacts] = {}
    for name, model_result in trained_models.items():
        results[name] = analyzer.analyze(
            model_name=name,
            model=model_result.best_estimator,
            X=X, y=y,
        )
    return results


if __name__ == "__main__":
    print("Módulo F3 (SHAP) cargado correctamente.")
