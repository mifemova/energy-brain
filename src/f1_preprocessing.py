"""
================================================================================
TFM - Aplicación de la IA en la Innovación Educativa
Universidad Internacional de La Rioja (UNIR)
Autores: Gloria Lida Alzate Suárez / Miguel Fernando Morales Valderrama
================================================================================

MÓDULO F1 — Preprocesamiento e Ingeniería de Características (OE1)

Este módulo implementa la primera fase del piloto experimental: la integración
de los siete archivos del Open University Learning Analytics Dataset (OULAD),
la codificación de la variable objetivo binaria, el tratamiento de valores
ausentes y la construcción del conjunto de características que alimentará a
los modelos de clasificación supervisada.

El diseño sigue el principio de reproducibilidad estricta: todas las operaciones
son deterministas (random_state fijado), idempotentes (el mismo input produce
siempre el mismo output) y observables (cada paso registra cuántos registros
procesa, cuántos descarta y por qué).

Nota metodológica clave: SMOTE NO se aplica en este módulo. El balanceo de
clases mediante SMOTE debe aplicarse exclusivamente sobre el conjunto de
entrenamiento de cada pliegue de la validación cruzada (ver módulo F2), para
evitar la filtración de información (data leakage) que inflaría artificialmente
las métricas de evaluación (Dong et al., 2025; Ovtšarenko, 2026).
================================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Configuración del registro de eventos del pipeline
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("F1_Preprocessing")

# Semilla global para garantizar reproducibilidad del experimento
RANDOM_STATE: int = 42


# ============================================================================
# 1. CONFIGURACIÓN Y ESQUEMA DE DATOS
# ============================================================================

@dataclass
class OULADSchema:
    """
    Esquema declarativo del Open University Learning Analytics Dataset.

    Documenta los siete archivos que componen el OULAD, las llaves primarias
    utilizadas para el merge relacional y las variables que se conservan tras
    el preprocesamiento. Tener este esquema centralizado en una sola clase
    facilita el mantenimiento del pipeline cuando aparecen nuevas versiones
    del dataset o se incorporan variables adicionales.
    """

    # Llaves primarias que identifican unívocamente a un estudiante en un curso
    primary_keys: tuple = ("id_student", "code_module", "code_presentation")

    # Archivos del OULAD y su propósito en el pipeline
    files: dict = field(default_factory=lambda: {
        "studentInfo": "Datos sociodemográficos y resultado final (target)",
        "studentRegistration": "Fechas de matrícula y baja",
        "studentAssessment": "Calificaciones obtenidas en cada evaluación",
        "assessments": "Catálogo de evaluaciones (tipo, peso, fecha límite)",
        "courses": "Catálogo de cursos y duración en días",
        "vle": "Catálogo de materiales del entorno virtual",
        "studentVle": "Interacciones estudiante-recurso (clave del análisis)",
    })

    # Variables sociodemográficas conservadas tras la codificación
    demographic_features: tuple = (
        "gender", "region", "highest_education", "imd_band",
        "age_band", "num_of_prev_attempts", "studied_credits", "disability",
    )

    # Mapeo de la variable objetivo: 1 = Withdrawn (abandono), 0 = resto
    target_mapping: dict = field(default_factory=lambda: {
        "Withdrawn": 1, "Fail": 0, "Pass": 0, "Distinction": 0,
    })


# ============================================================================
# 2. CARGA E INTEGRACIÓN DE LOS ARCHIVOS DEL OULAD
# ============================================================================

class OULADLoader:
    """
    Carga los siete archivos CSV del OULAD desde una ruta local.

    El OULAD se distribuye como un archivo ZIP en https://analyse.kmi.open.ac.uk/.
    Para reproducir el experimento, descarga el dataset, descomprímelo en una
    carpeta y pasa esa carpeta como argumento al constructor.

    Alternativamente, el dataset puede obtenerse desde Kaggle usando la CLI
    oficial:
        kaggle datasets download -d anlgrbz/student-demographics-online-education-dataoulad
    """

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        self.schema = OULADSchema()
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"La ruta del dataset no existe: {self.data_path}. "
                "Descarga el OULAD desde https://analyse.kmi.open.ac.uk/ o "
                "mediante la CLI de Kaggle: kaggle datasets download -d "
                "anlgrbz/student-demographics-online-education-dataoulad."
            )

    def load_all(self) -> dict[str, pd.DataFrame]:
        """
        Carga cada archivo del OULAD en un DataFrame de Pandas.

        Devuelve un diccionario {nombre_archivo: DataFrame} para acceso por
        nombre. Cada DataFrame conserva las columnas originales del OULAD;
        las transformaciones se aplican en los pasos siguientes.
        """
        tables: dict[str, pd.DataFrame] = {}
        for table_name in self.schema.files:
            csv_path = self.data_path / f"{table_name}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Falta el archivo {csv_path}")
            df = pd.read_csv(csv_path)
            tables[table_name] = df
            logger.info(
                "Cargado %s.csv → %d filas × %d columnas",
                table_name, len(df), len(df.columns),
            )
        return tables


# ============================================================================
# 3. INGENIERÍA DE CARACTERÍSTICAS DE INTERACCIÓN CON EL LMS
# ============================================================================

class LMSFeatureBuilder:
    """
    Construye variables agregadas a partir del registro de interacciones LMS.

    El archivo studentVle.csv del OULAD registra cada clic que un estudiante
    realiza sobre un material del entorno virtual de aprendizaje, con su fecha
    y el número de clics emitidos en ese día. La granularidad es excesiva para
    alimentar directamente a un clasificador, por lo que este componente agrega
    la información a nivel (estudiante, módulo, presentación), produciendo
    variables interpretables pedagógicamente:

        - Volumen total de actividad (suma y mediana de clics).
        - Regularidad temporal (número de días con actividad).
        - Actividad temprana (clics en las primeras dos semanas), variable
          clave para la detección temprana del riesgo de abandono.
        - Distribución de la actividad por tipo de recurso (foro, contenido,
          quiz, etc.), informativa sobre el patrón de uso del LMS.

    La selección de estas variables se fundamenta en la evidencia de Ovtšarenko
    (2026) sobre la primacía de las variables de interacción y planificación
    temporal en la predicción del éxito en aprendizaje en línea.
    """

    EARLY_WEEKS_DAYS: int = 14  # Ventana temporal para detección temprana

    def build_aggregate_features(
        self,
        student_vle: pd.DataFrame,
        vle: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Agrega las interacciones del LMS por (estudiante, módulo, presentación).

        Parámetros:
            student_vle : registros brutos de interacción estudiante-recurso.
            vle         : catálogo de materiales con su tipo (activity_type).

        Devuelve un DataFrame con las variables agregadas listas para el merge.
        """
        # Enriquecemos las interacciones con el tipo de recurso del catálogo VLE
        df = student_vle.merge(
            vle[["id_site", "code_module", "code_presentation", "activity_type"]],
            on=["id_site", "code_module", "code_presentation"],
            how="left",
        )

        # Métricas generales de volumen y regularidad
        general = (
            df.groupby(list(OULADSchema().primary_keys), observed=True)
              .agg(
                  total_clicks=("sum_click", "sum"),
                  median_clicks_per_day=("sum_click", "median"),
                  interaction_days=("date", "nunique"),
                  first_interaction_day=("date", "min"),
                  last_interaction_day=("date", "max"),
              )
              .reset_index()
        )

        # Actividad en la ventana temprana del curso (detección temprana)
        early = df[df["date"] <= self.EARLY_WEEKS_DAYS]
        early_features = (
            early.groupby(list(OULADSchema().primary_keys), observed=True)
                 .agg(
                     early_clicks=("sum_click", "sum"),
                     early_active_days=("date", "nunique"),
                 )
                 .reset_index()
        )

        # Distribución de clics por tipo de recurso del LMS
        # Limitamos a las cinco actividades más frecuentes del OULAD para evitar
        # la explosión dimensional (los tipos minoritarios añaden ruido sin
        # aportar información discriminante).
        top_activities = ["forumng", "oucontent", "quiz", "resource", "url"]
        activity_pivot = (
            df[df["activity_type"].isin(top_activities)]
              .groupby(list(OULADSchema().primary_keys) + ["activity_type"],
                       observed=True)["sum_click"].sum()
              .unstack(fill_value=0)
              .reset_index()
              .rename(columns={a: f"clicks_{a}" for a in top_activities})
        )

        # Merge progresivo de las tres tablas de variables LMS
        features = general.merge(
            early_features, on=list(OULADSchema().primary_keys), how="left",
        ).merge(
            activity_pivot, on=list(OULADSchema().primary_keys), how="left",
        )

        # Los estudiantes sin actividad en la ventana temprana presentan NaN.
        # Codificamos esa ausencia explícita como cero clics, manteniendo así
        # la información de "no hubo interacción temprana", que es justamente
        # una señal de riesgo relevante (no un dato ausente que deba imputarse).
        early_cols = ["early_clicks", "early_active_days"] + \
                     [f"clicks_{a}" for a in top_activities]
        features[early_cols] = features[early_cols].fillna(0)

        logger.info(
            "Variables LMS construidas: %d estudiantes × %d variables",
            len(features), len(features.columns),
        )
        return features


# ============================================================================
# 4. INGENIERÍA DE CARACTERÍSTICAS DE RENDIMIENTO EVALUATIVO
# ============================================================================

class AssessmentFeatureBuilder:
    """
    Construye variables agregadas de rendimiento en evaluaciones.

    El archivo studentAssessment.csv registra cada calificación obtenida por
    un estudiante en cada evaluación de un curso. Combinado con el catálogo
    assessments.csv (que incluye el peso de cada evaluación en la calificación
    final), permite construir indicadores ponderados de rendimiento que actúan
    como predictores complementarios a las variables de interacción del LMS.
    """

    def build_features(
        self,
        student_assessment: pd.DataFrame,
        assessments: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calcula puntaje medio, mediana, desviación y número de evaluaciones
        entregadas por (estudiante, módulo, presentación).
        """
        # Unimos las calificaciones con el catálogo para conocer peso y tipo
        df = student_assessment.merge(
            assessments[["id_assessment", "code_module", "code_presentation",
                         "assessment_type", "weight"]],
            on="id_assessment",
            how="left",
        )

        agg = (
            df.groupby(list(OULADSchema().primary_keys), observed=True)
              .agg(
                  mean_score=("score", "mean"),
                  median_score=("score", "median"),
                  std_score=("score", "std"),
                  num_assessments_submitted=("id_assessment", "nunique"),
              )
              .reset_index()
        )

        # std produce NaN cuando hay una sola evaluación; lo codificamos como 0
        agg["std_score"] = agg["std_score"].fillna(0)
        logger.info("Variables de evaluación construidas para %d filas", len(agg))
        return agg


# ============================================================================
# 5. ORQUESTADOR DEL PREPROCESAMIENTO
# ============================================================================

@dataclass
class PreprocessingArtifacts:
    """
    Empaqueta los productos del preprocesamiento para uso en fases posteriores.
    Mantener un objeto explícito (en lugar de tuplas anónimas) facilita la
    documentación y reduce la probabilidad de error al consumir el pipeline.
    """
    X: pd.DataFrame           # Matriz de características procesada
    y: pd.Series              # Variable objetivo binaria (1 = Withdrawn)
    feature_names: list[str]  # Nombres de las columnas tras codificación
    class_balance: dict       # Distribución original de la clase positiva


class OULADPreprocessor:
    """
    Orquestador del flujo de preprocesamiento completo para el OULAD.

    Encadena las siguientes operaciones:

        (1) Carga de los siete archivos del OULAD.
        (2) Construcción de las variables LMS agregadas (LMSFeatureBuilder).
        (3) Construcción de las variables de evaluación (AssessmentFeatureBuilder).
        (4) Merge con la tabla studentInfo y codificación de la variable objetivo.
        (5) Imputación de valores ausentes (mediana para numéricas, moda para
            categóricas) y codificación one-hot de las variables nominales.

    El resultado es una matriz X y un vector y listos para alimentar a la
    fase F2 de entrenamiento.
    """

    def __init__(self, data_path: str | Path):
        self.loader = OULADLoader(data_path)
        self.lms_builder = LMSFeatureBuilder()
        self.assessment_builder = AssessmentFeatureBuilder()
        self.schema = OULADSchema()

    def run(self) -> PreprocessingArtifacts:
        """Ejecuta el pipeline de preprocesamiento de principio a fin."""

        # ── (1) Carga ────────────────────────────────────────────────────
        tables = self.loader.load_all()

        # ── (2) Variables LMS ────────────────────────────────────────────
        lms_features = self.lms_builder.build_aggregate_features(
            tables["studentVle"], tables["vle"],
        )

        # ── (3) Variables de evaluación ──────────────────────────────────
        assessment_features = self.assessment_builder.build_features(
            tables["studentAssessment"], tables["assessments"],
        )

        # ── (4) Merge con studentInfo y codificación del target ──────────
        df = tables["studentInfo"].copy()
        df["target"] = df["final_result"].map(self.schema.target_mapping)

        if df["target"].isna().any():
            raise ValueError("Hay valores de 'final_result' fuera del mapeo conocido.")

        df = df.merge(lms_features, on=list(self.schema.primary_keys), how="left")
        df = df.merge(assessment_features, on=list(self.schema.primary_keys), how="left")

        # Tras los left-join, los estudiantes sin interacciones LMS o sin
        # evaluaciones entregadas presentan NaN en las columnas correspondientes.
        # Para las variables de conteo, NaN equivale a "cero actividad" y se
        # imputa como tal; para mean_score y median_score, la mediana refleja
        # mejor el supuesto de "rendimiento desconocido pero típico".

        # ── (5) Imputación ──────────────────────────────────────────────
        numeric_count_cols = [
            "total_clicks", "median_clicks_per_day", "interaction_days",
            "first_interaction_day", "last_interaction_day",
            "early_clicks", "early_active_days",
            "clicks_forumng", "clicks_oucontent", "clicks_quiz",
            "clicks_resource", "clicks_url",
            "num_assessments_submitted",
        ]
        df[numeric_count_cols] = df[numeric_count_cols].fillna(0)

        score_cols = ["mean_score", "median_score", "std_score"]
        for col in score_cols:
            df[col] = df[col].fillna(df[col].median())

        # Imputación por moda para categóricas (imd_band es la más afectada
        # por NaN en el OULAD: aproximadamente el 3 % de los registros).
        for col in self.schema.demographic_features:
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode().iloc[0])

        # ── Codificación one-hot de las variables categóricas ───────────
        categorical_cols = [
            c for c in self.schema.demographic_features
            if df[c].dtype == "object"
        ]
        df_encoded = pd.get_dummies(
            df, columns=categorical_cols, drop_first=True, dtype=int,
        )

        # Separación entre matriz de características y variable objetivo.
        # Eliminamos columnas que filtrarían información del target (final_result)
        # o que no son predictoras (identificadores).
        cols_to_drop = [
            "final_result", "target",
            "id_student", "code_module", "code_presentation",
        ]
        X = df_encoded.drop(columns=cols_to_drop)
        y = df_encoded["target"].astype(int)

        # Sanitize column names for XGBoost compatibility (no [, ], <, >, =)
        X.columns = [
            col.replace('[', '_').replace(']', '_').replace('<', '_lt_').replace('>', '_gt_').replace('=', '_eq_')
            for col in X.columns
        ]

        # Diagnóstico del desbalance de clases (insumo crítico para F2)
        n_pos = int(y.sum())
        n_neg = int((y == 0).sum())
        class_balance = {
            "n_total": int(len(y)),
            "n_withdrawn": n_pos,
            "n_not_withdrawn": n_neg,
            "withdrawn_pct": round(100 * n_pos / len(y), 2),
        }
        logger.info("Distribución de clases → %s", class_balance)

        return PreprocessingArtifacts(
            X=X,
            y=y,
            feature_names=list(X.columns),
            class_balance=class_balance,
        )


# ============================================================================
# Ejemplo de uso (cuando se ejecuta el módulo directamente)
# ============================================================================

if __name__ == "__main__":
    # Ajusta esta ruta a la ubicación de los CSV del OULAD en tu sistema
    OULAD_PATH = Path("./data/oulad")

    preprocessor = OULADPreprocessor(OULAD_PATH)
    artifacts = preprocessor.run()

    print(f"\n[F1] Matriz de características: {artifacts.X.shape}")
    print(f"[F1] Balance de clases: {artifacts.class_balance}")
    print(f"[F1] Variables construidas: {len(artifacts.feature_names)}")
