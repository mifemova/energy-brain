"""
================================================================================
MÓDULO F4 — Análisis Estadístico de Consistencia (OE4)
================================================================================

Este módulo valida estadísticamente si los predictores identificados por
SHAP son robustos (estables entre los tres algoritmos) o si dependen del
método de modelado utilizado. La decisión de usar pruebas no paramétricas
(Kruskal-Wallis y Mann-Whitney U) se justifica por dos razones:

    1. Las distribuciones de valores SHAP por variable suelen ser fuertemente
       sesgadas y multimodales, vulnerando los supuestos de normalidad y
       homocedasticidad que requieren las pruebas paramétricas (ANOVA, t de
       Student).

    2. El interés inferencial recae sobre la mediana, no sobre la media: una
       variable es un "predictor robusto" si la distribución central de su
       contribución SHAP es similar entre algoritmos, con independencia de
       eventuales outliers.

La corrección de Bonferroni controla el error de tipo I cuando se realizan
múltiples comparaciones simultáneas. Para un conjunto de k variables y
tres algoritmos, se ejecutan k × 3 pruebas Kruskal-Wallis y, cuando esta
detecta diferencias, k × 3 comparaciones pareadas Mann-Whitney U. El
umbral alpha global de 0.05 se divide por el número total de pruebas,
manteniendo la tasa de error familiar en el nivel nominal.

Criterio de robustez adoptado:
    Una variable se clasifica como PREDICTOR ROBUSTO si la prueba de
    Kruskal-Wallis NO detecta diferencias significativas entre las
    distribuciones SHAP de los tres modelos (p > alpha corregido). Esto
    indica que el poder predictivo de esa variable es transferible entre
    algoritmos, no un artefacto del método de modelado.
================================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("F4_Statistics")


# ============================================================================
# 1. ESTRUCTURAS DE RESULTADOS
# ============================================================================

@dataclass
class ConsistencyResults:
    """Empaqueta los resultados del análisis estadístico de consistencia."""
    kruskal_results: pd.DataFrame          # Una fila por variable analizada
    pairwise_results: pd.DataFrame         # Comparaciones por pares Mann-Whitney
    robust_predictors: list[str]           # Variables sin diferencia significativa
    algorithm_dependent: list[str]         # Variables con diferencia significativa
    alpha_bonferroni: float                # Umbral corregido aplicado


# ============================================================================
# 2. PRUEBA DE KRUSKAL-WALLIS POR VARIABLE
# ============================================================================

class SHAPConsistencyAnalyzer:
    """
    Compara las distribuciones de importancia SHAP entre los tres modelos.

    El análisis se realiza variable a variable: para cada variable, se extrae
    el vector de valores SHAP que cada modelo asignó a esa variable a lo
    largo de todas las muestras analizadas. La prueba de Kruskal-Wallis
    contrasta la hipótesis nula de que las tres distribuciones (una por
    modelo) provienen de la misma población.
    """

    def __init__(self, alpha: float = 0.05, top_n: int | None = 20):
        """
        Parámetros:
            alpha : nivel de significación global del experimento.
            top_n : número de variables más importantes a comparar. Si es
                    None, se comparan todas las variables del dataset. Para
                    el TFM se recomienda usar las 15-25 variables con mayor
                    importancia media SHAP entre los tres modelos, pues
                    son las que efectivamente conducen las predicciones.
        """
        self.alpha = alpha
        self.top_n = top_n

    def _select_variables(
        self,
        shap_dict: dict[str, np.ndarray],
        feature_names: list[str],
    ) -> list[str]:
        """Selecciona las top_n variables más importantes globalmente."""
        if self.top_n is None:
            return feature_names
        # Importancia agregada: media de las medias absolutas entre modelos
        importances = []
        for arr in shap_dict.values():
            importances.append(np.abs(arr).mean(axis=0))
        global_imp = np.mean(importances, axis=0)
        order = np.argsort(global_imp)[::-1][:self.top_n]
        return [feature_names[i] for i in order]

    def analyze(
        self,
        shap_dict: dict[str, np.ndarray],
        feature_names: list[str],
    ) -> ConsistencyResults:
        """
        Ejecuta el análisis de consistencia.

        Parámetros:
            shap_dict : {nombre_modelo: array_shap_values (n_muestras x n_vars)}
            feature_names : nombres de las variables (deben coincidir entre modelos)

        Devuelve un ConsistencyResults con los resultados completos.
        """
        if len(shap_dict) < 2:
            raise ValueError("Se requieren al menos dos modelos para comparar.")

        # Verificamos que todos los modelos analizaron el mismo número de muestras
        sample_sizes = {name: arr.shape[0] for name, arr in shap_dict.items()}
        if len(set(sample_sizes.values())) > 1:
            logger.warning(
                "Tamaños de muestra distintos entre modelos: %s. "
                "Se ajustará al mínimo común.", sample_sizes,
            )
            min_n = min(sample_sizes.values())
            shap_dict = {n: a[:min_n] for n, a in shap_dict.items()}

        selected_vars = self._select_variables(shap_dict, feature_names)
        var_index = {n: i for i, n in enumerate(feature_names)}
        models = list(shap_dict.keys())

        # Número total de pruebas para la corrección de Bonferroni
        n_kruskal_tests = len(selected_vars)
        n_pairs = len(models) * (len(models) - 1) // 2
        n_pairwise_tests = n_kruskal_tests * n_pairs
        # Aplicamos la corrección al total combinado de pruebas
        alpha_bonferroni = self.alpha / (n_kruskal_tests + n_pairwise_tests)
        logger.info(
            "Corrección de Bonferroni: alpha = %.6f (%d pruebas totales)",
            alpha_bonferroni, n_kruskal_tests + n_pairwise_tests,
        )

        # ── Kruskal-Wallis variable a variable ────────────────────────────
        kruskal_rows = []
        pairwise_rows = []
        robust_predictors: list[str] = []
        algorithm_dependent: list[str] = []

        for var in selected_vars:
            j = var_index[var]
            samples = [shap_dict[m][:, j] for m in models]

            # Aplicamos Kruskal-Wallis; si todas las muestras son idénticas,
            # devolvemos p = 1 manualmente para evitar la advertencia de SciPy.
            try:
                h_stat, p_value = stats.kruskal(*samples)
            except ValueError:
                h_stat, p_value = 0.0, 1.0

            is_significant = bool(p_value < alpha_bonferroni)
            kruskal_rows.append({
                "variable": var,
                "H_statistic": float(h_stat),
                "p_value": float(p_value),
                "p_value_significativo_Bonferroni": is_significant,
                "mediana_SHAP_por_modelo": {
                    m: float(np.median(shap_dict[m][:, j])) for m in models
                },
            })

            if is_significant:
                algorithm_dependent.append(var)
                # Mann-Whitney U con corrección Bonferroni para identificar
                # qué par(es) de modelos difieren
                for i, m1 in enumerate(models):
                    for m2 in models[i + 1:]:
                        u_stat, p_pair = stats.mannwhitneyu(
                            shap_dict[m1][:, j],
                            shap_dict[m2][:, j],
                            alternative="two-sided",
                        )
                        pairwise_rows.append({
                            "variable": var,
                            "modelo_A": m1,
                            "modelo_B": m2,
                            "U_statistic": float(u_stat),
                            "p_value": float(p_pair),
                            "p_significativo_Bonferroni": bool(p_pair < alpha_bonferroni),
                        })
            else:
                robust_predictors.append(var)

        return ConsistencyResults(
            kruskal_results=pd.DataFrame(kruskal_rows),
            pairwise_results=pd.DataFrame(pairwise_rows),
            robust_predictors=robust_predictors,
            algorithm_dependent=algorithm_dependent,
            alpha_bonferroni=alpha_bonferroni,
        )


# ============================================================================
# 3. INTERPRETACIÓN PEDAGÓGICA DE LOS RESULTADOS
# ============================================================================

def summarize_consistency(results: ConsistencyResults) -> str:
    """
    Produce un resumen en prosa académica de los resultados de consistencia,
    apto para incluirse directamente en el Capítulo 5 (Resultados) del TFM.
    """
    total = len(results.robust_predictors) + len(results.algorithm_dependent)
    if total == 0:
        return "No se analizaron variables; el resultado está vacío."

    pct_robust = 100 * len(results.robust_predictors) / total

    text = (
        f"Tras aplicar la prueba de Kruskal-Wallis con corrección de Bonferroni "
        f"(α corregido = {results.alpha_bonferroni:.5f}) sobre las {total} "
        f"variables de mayor importancia SHAP, {len(results.robust_predictors)} "
        f"variables ({pct_robust:.1f} %) presentan distribuciones equivalentes "
        f"entre los tres algoritmos y se clasifican como predictores robustos: "
        f"{', '.join(results.robust_predictors[:10])}"
        + (' …' if len(results.robust_predictors) > 10 else '') + ". "
        f"Las {len(results.algorithm_dependent)} variables restantes presentan "
        f"diferencias significativas entre al menos dos modelos, lo que indica "
        f"que su poder predictivo depende parcialmente del algoritmo. La tabla "
        f"de comparaciones pareadas Mann-Whitney U detalla qué pares de "
        f"modelos divergen en cada caso."
    )
    return text


if __name__ == "__main__":
    print("Módulo F4 (consistencia estadística) cargado correctamente.")
