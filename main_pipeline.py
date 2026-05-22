"""
================================================================================
TFM - Aplicación de la IA en la Innovación Educativa
Universidad Internacional de La Rioja (UNIR)
================================================================================

main_pipeline.py — Orquestador principal del piloto experimental

Este script ejecuta de principio a fin las cinco fases (F1-F5) del piloto.
Está diseñado para correr desde la línea de comandos o desde un notebook,
y produce todos los artefactos requeridos por la memoria del TFM:

    - results/preprocessed_dataset.parquet   : matriz X procesada
    - results/comparison_table.csv           : Tabla comparativa de modelos
    - results/cv_per_fold_metrics.csv        : métricas por pliegue
    - results/shap_summary_<modelo>.png      : summary plots SHAP
    - results/shap_dependence_<modelo>_*.png : dependence plots SHAP
    - results/shap_waterfall_<modelo>_*.png  : waterfall plots SHAP
    - results/consistency_kruskal.csv        : pruebas Kruskal-Wallis
    - results/consistency_pairwise.csv       : Mann-Whitney U corregido
    - results/risk_recommendations.json      : catálogo de recomendaciones
    - results/likert_questionnaire.csv       : cuestionario para tutores
    - results/dashboard_data.json            : datos consumidos por el dashboard

Uso:
    python main_pipeline.py --data-path ./data/oulad --output-dir ./results
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Importación de los módulos del pipeline
from src.f1_preprocessing import OULADPreprocessor
from src.f2_modeling import ModelComparator
from src.f3_shap_analysis import run_shap_analysis_all_models
from src.f4_statistical_consistency import (
    SHAPConsistencyAnalyzer, summarize_consistency,
)
from src.f5_pedagogy import (
    get_baseline_risk_profiles, build_validation_questionnaire,
    classify_students_by_risk,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MainPipeline")


# ============================================================================
# UTILIDAD: serialización segura de objetos NumPy a JSON
# ============================================================================

def _to_json_safe(obj):
    """Convierte arrays NumPy y tipos no-serializables a primitivos."""
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if hasattr(obj, "__dataclass_fields__"):
        return _to_json_safe(asdict(obj))
    return obj


# ============================================================================
# ORQUESTADOR PRINCIPAL
# ============================================================================

class Pipeline:
    """Ejecuta secuencialmente las cinco fases del piloto experimental."""

    def __init__(
        self,
        data_path: str | Path,
        output_dir: str | Path,
        shap_sample_size: int = 3000,
        n_outer_folds: int = 10,
        n_inner_folds: int = 3,
    ):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shap_sample_size = shap_sample_size
        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds

    # ── F1 ─────────────────────────────────────────────────────────────
    def run_phase_1(self):
        logger.info("════════════ FASE 1: Preprocesamiento (OE1) ════════════")
        preprocessor = OULADPreprocessor(self.data_path)
        artifacts = preprocessor.run()

        artifacts.X.to_parquet(self.output_dir / "preprocessed_features.parquet")
        artifacts.y.to_frame().to_parquet(self.output_dir / "preprocessed_target.parquet")

        with open(self.output_dir / "class_balance.json", "w", encoding="utf-8") as f:
            json.dump(artifacts.class_balance, f, indent=2, ensure_ascii=False)
        return artifacts

    # ── F2 ─────────────────────────────────────────────────────────────
    def run_phase_2(self, X: pd.DataFrame, y: pd.Series):
        logger.info("════════════ FASE 2: Entrenamiento (OE2) ════════════")
        comparator = ModelComparator(
            n_outer_folds=self.n_outer_folds,
            n_inner_folds=self.n_inner_folds,
        )
        model_results = comparator.run_all(X, y)

        # Tabla comparativa
        comparison = comparator.build_comparison_table(model_results)
        comparison.to_csv(self.output_dir / "comparison_table.csv", index=False)
        logger.info("\n%s", comparison.to_string(index=False))

        # Métricas por pliegue de cada modelo
        for name, r in model_results.items():
            r.per_fold_metrics.to_csv(
                self.output_dir / f"cv_per_fold_{name}.csv", index=False,
            )
            joblib.dump(
                r.best_estimator,
                self.output_dir / f"model_{name}.joblib",
            )
        return model_results

    # ── F3 ─────────────────────────────────────────────────────────────
    def run_phase_3(self, model_results, X, y):
        logger.info("════════════ FASE 3: Análisis SHAP (OE3) ════════════")
        shap_results = run_shap_analysis_all_models(
            trained_models=model_results,
            X=X, y=y,
            output_dir=self.output_dir,
            sample_size=self.shap_sample_size,
        )
        for name, r in shap_results.items():
            r.global_importance.to_csv(
                self.output_dir / f"shap_global_importance_{name}.csv",
                header=["mean_abs_shap"],
            )
        return shap_results

    # ── F4 ─────────────────────────────────────────────────────────────
    def run_phase_4(self, shap_results):
        logger.info("════════════ FASE 4: Análisis de consistencia (OE4) ════════════")

        # Construimos el diccionario {modelo: array_shap} usando solo las
        # variables comunes a todos los modelos (que en nuestro caso es la
        # misma matriz X, por lo tanto coinciden completamente).
        any_artifacts = next(iter(shap_results.values()))
        feature_names = any_artifacts.feature_names
        shap_dict = {
            name: art.shap_values for name, art in shap_results.items()
        }

        analyzer = SHAPConsistencyAnalyzer(alpha=0.05, top_n=20)
        consistency = analyzer.analyze(shap_dict, feature_names)

        consistency.kruskal_results.to_csv(
            self.output_dir / "consistency_kruskal.csv", index=False,
        )
        consistency.pairwise_results.to_csv(
            self.output_dir / "consistency_pairwise.csv", index=False,
        )
        with open(self.output_dir / "robust_predictors.json", "w",
                  encoding="utf-8") as f:
            json.dump({
                "robust_predictors": consistency.robust_predictors,
                "algorithm_dependent": consistency.algorithm_dependent,
                "alpha_bonferroni": consistency.alpha_bonferroni,
                "summary": summarize_consistency(consistency),
            }, f, indent=2, ensure_ascii=False)

        logger.info(summarize_consistency(consistency))
        return consistency

    # ── F5 ─────────────────────────────────────────────────────────────
    def run_phase_5(self, best_model, X):
        logger.info("════════════ FASE 5: Validación pedagógica (OE5) ════════════")
        profiles = get_baseline_risk_profiles()

        # Predicción de probabilidades sobre el dataset completo
        y_proba = best_model.predict_proba(X)[:, 1]
        risk_labels = classify_students_by_risk(y_proba, profiles)

        # Distribución de estudiantes por nivel de riesgo
        risk_distribution = (
            risk_labels.value_counts(normalize=True)
            .round(3).to_dict()
        )

        # Persistencia
        recommendations = {name: asdict(p) for name, p in profiles.items()}
        with open(self.output_dir / "risk_recommendations.json", "w",
                  encoding="utf-8") as f:
            json.dump({
                "profiles": _to_json_safe(recommendations),
                "risk_distribution": _to_json_safe(risk_distribution),
            }, f, indent=2, ensure_ascii=False)

        questionnaire = build_validation_questionnaire(profiles)
        questionnaire.to_csv(
            self.output_dir / "likert_questionnaire.csv",
            index=False, encoding="utf-8",
        )

        return profiles, risk_labels, risk_distribution

    # ── EXPORTACIÓN PARA EL DASHBOARD ──────────────────────────────────
    def export_dashboard_data(
        self, artifacts, model_results, shap_results, consistency,
        profiles, risk_distribution,
    ):
        """Genera un único JSON consumido por el dashboard interactivo."""
        comparison_table = ModelComparator.build_comparison_table(model_results)
        any_artifacts = next(iter(shap_results.values()))

        dashboard_data = {
            "class_balance": artifacts.class_balance,
            "comparison_table": comparison_table.to_dict(orient="records"),
            "per_fold_metrics": {
                name: r.per_fold_metrics.to_dict(orient="records")
                for name, r in model_results.items()
            },
            "global_shap_importance": {
                name: r.global_importance.head(15).to_dict()
                for name, r in shap_results.items()
            },
            "kruskal_results": consistency.kruskal_results.to_dict(orient="records"),
            "robust_predictors": consistency.robust_predictors,
            "algorithm_dependent": consistency.algorithm_dependent,
            "risk_distribution": risk_distribution,
            "profiles": {
                name: {
                    "level": p.level,
                    "probability_range": list(p.probability_range),
                    "typical_signals": p.typical_signals,
                    "recommended_actions": p.recommended_actions,
                    "responsible_actors": p.responsible_actors,
                    "time_horizon": p.time_horizon,
                }
                for name, p in profiles.items()
            },
            "feature_names": any_artifacts.feature_names,
        }

        out_path = self.output_dir / "dashboard_data.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(_to_json_safe(dashboard_data), f, indent=2, ensure_ascii=False)
        logger.info("Datos del dashboard exportados a %s", out_path)

    # ── EJECUCIÓN COMPLETA ─────────────────────────────────────────────
    def run(self):
        """Ejecuta el pipeline de extremo a extremo."""
        artifacts = self.run_phase_1()
        model_results = self.run_phase_2(artifacts.X, artifacts.y)
        shap_results = self.run_phase_3(model_results, artifacts.X, artifacts.y)
        consistency = self.run_phase_4(shap_results)

        # Selección del modelo de mejor AUC-ROC para la fase pedagógica
        best_name = max(
            model_results,
            key=lambda n: model_results[n].cv_metrics["auc_roc_mean"],
        )
        logger.info("Mejor modelo según AUC-ROC: %s", best_name)
        profiles, risk_labels, risk_dist = self.run_phase_5(
            model_results[best_name].best_estimator, artifacts.X,
        )

        self.export_dashboard_data(
            artifacts, model_results, shap_results, consistency,
            profiles, risk_dist,
        )

        logger.info("════════════ PIPELINE COMPLETADO ════════════")
        return {
            "artifacts": artifacts,
            "model_results": model_results,
            "shap_results": shap_results,
            "consistency": consistency,
            "profiles": profiles,
            "risk_distribution": risk_dist,
            "best_model_name": best_name,
        }


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline experimental del TFM (UNIR) — OULAD + SHAP",
    )
    parser.add_argument(
        "--data-path", default="./data/oulad", type=str,
        help="Ruta a los CSV del OULAD (sin comprimir).",
    )
    parser.add_argument(
        "--output-dir", default="./results", type=str,
        help="Directorio donde se guardarán los artefactos.",
    )
    parser.add_argument(
        "--shap-sample-size", default=3000, type=int,
        help="Muestra estratificada para el cálculo SHAP.",
    )
    parser.add_argument(
        "--n-outer-folds", default=10, type=int,
        help="Pliegues externos para la validación cruzada estratificada.",
    )
    parser.add_argument(
        "--n-inner-folds", default=3, type=int,
        help="Pliegues internos para el GridSearchCV anidado.",
    )
    args = parser.parse_args()

    pipeline = Pipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        shap_sample_size=args.shap_sample_size,
        n_outer_folds=args.n_outer_folds,
        n_inner_folds=args.n_inner_folds,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
