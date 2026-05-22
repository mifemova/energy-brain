"""
================================================================================
MÓDULO F5 — Formulación y Validación Pedagógica (OE5)
================================================================================

Este módulo cierra el ciclo del piloto traduciendo los hallazgos técnicos en
recomendaciones pedagógicas accionables. La traducción no es trivial: requiere
mapear cada señal cuantitativa del LMS (por ejemplo, "total_clicks < 50 en las
primeras dos semanas") a un texto comprensible para un tutor que no es
ingeniero de datos ("el estudiante presenta una baja interacción inicial con
el aula virtual, lo que constituye una señal temprana de desvinculación").

La estructura adoptada organiza las recomendaciones en tres niveles de riesgo,
derivados de los percentiles de la probabilidad predicha por el mejor modelo:

    - RIESGO ALTO   (probabilidad > percentil 75): intervención urgente.
    - RIESGO MEDIO  (percentil 50–75): seguimiento intensificado.
    - RIESGO BAJO   (percentil < 50): refuerzo motivacional preventivo.

Cada nivel se acompaña de:
    (1) las señales SHAP características del perfil,
    (2) las acciones pedagógicas recomendadas,
    (3) los actores institucionales responsables,
    (4) un horizonte temporal sugerido.

El protocolo de validación está diseñado para aplicarse con un mínimo de tres
tutores expertos en educación virtual, evaluando seis dimensiones en escala
Likert de 1 a 5: comprensibilidad, aplicabilidad, pertinencia, novedad,
factibilidad e impacto percibido.
================================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger("F5_Pedagogy")


# ============================================================================
# 1. PERFIL DE RIESGO Y RECOMENDACIONES
# ============================================================================

@dataclass
class RiskProfile:
    """
    Describe un perfil de riesgo y las recomendaciones asociadas.

    Esta estructura es la unidad mínima del catálogo pedagógico que se entrega
    a los tutores. La separación entre 'señales' (la evidencia que justifica
    la clasificación) y 'acciones' (lo que el tutor debe hacer) es deliberada:
    promueve la transparencia algorítmica que la literatura identifica como
    requisito de legitimidad pedagógica (Ayala Sánchez, 2025).
    """
    level: str                           # "alto" | "medio" | "bajo"
    probability_range: tuple[float, float]
    typical_signals: list[str]           # Lecturas SHAP características
    recommended_actions: list[str]       # Intervenciones pedagógicas
    responsible_actors: list[str]
    time_horizon: str
    rationale: str                       # Justificación basada en literatura


# ============================================================================
# 2. CATÁLOGO BASE DE RECOMENDACIONES PEDAGÓGICAS
# ============================================================================

def get_baseline_risk_profiles() -> dict[str, RiskProfile]:
    """
    Devuelve el catálogo inicial de recomendaciones, antes de la validación
    con tutores. Este catálogo se construye sobre la evidencia del estado del
    arte revisado en el Capítulo 2 y se refinará iterativamente tras la
    consulta a los expertos (paso de validación cualitativa).
    """
    return {
        "alto": RiskProfile(
            level="alto",
            probability_range=(0.75, 1.0),
            typical_signals=[
                "Total de clics en el LMS en el cuartil inferior del curso",
                "Cero o muy pocos accesos al aula virtual en las primeras dos semanas",
                "Calificación media en evaluaciones por debajo del percentil 25",
                "Ausencia de interacción en foros (clicks_forumng = 0)",
                "Una o más evaluaciones no entregadas",
            ],
            recommended_actions=[
                "Contacto sincrónico individualizado del tutor en menos de 48 horas, "
                "preferiblemente videollamada para verificar la situación del estudiante.",
                "Diagnóstico estructurado de las causas del bajo compromiso "
                "(motivacionales, técnicas, socioeconómicas o personales).",
                "Activación del protocolo institucional de retención: derivación a "
                "los servicios de orientación psicopedagógica y financiera si procede.",
                "Plan de recuperación académica con hitos semanales y compromiso "
                "explícito del estudiante, registrado en el expediente.",
                "Seguimiento semanal del nivel de actividad en el LMS hasta normalizar.",
            ],
            responsible_actors=["Tutor académico", "Coordinador de programa",
                                "Servicio de orientación estudiantil"],
            time_horizon="Intervención dentro de las primeras 48 horas; seguimiento "
                         "durante las cuatro semanas siguientes.",
            rationale=(
                "Las cifras de Pizzatto Blow et al. (2024) y Cao y Mai (2025) "
                "sitúan la deserción virtual entre el 40 % y el 80 %. Los "
                "modelos de alerta temprana documentados en el e-book Modelos "
                "predictivos REDUNIR (2024) reducen los falsos negativos hasta "
                "un 22 % cuando la intervención se produce en las dos primeras "
                "semanas, ventana coincidente con la variable early_clicks "
                "construida en el pipeline."
            ),
        ),

        "medio": RiskProfile(
            level="medio",
            probability_range=(0.50, 0.75),
            typical_signals=[
                "Interacción LMS irregular: días con alta actividad seguidos de "
                "lapsos prolongados sin acceso.",
                "Calificación media de evaluaciones entre los percentiles 25 y 50.",
                "Tendencia descendente del número de clics semana a semana.",
                "Participación esporádica en foros (clicks_forumng > 0 pero baja).",
            ],
            recommended_actions=[
                "Mensaje proactivo del tutor mediante el sistema de comunicación "
                "del LMS, recordando los próximos hitos del curso.",
                "Invitación a sesión grupal de tutoría sincrónica con compañeros "
                "del mismo perfil de actividad, fomentando la integración social.",
                "Sugerencia de recursos complementarios específicos según las "
                "evaluaciones con menor rendimiento.",
                "Configuración de notificaciones automáticas en el LMS para "
                "recordatorios de entrega.",
            ],
            responsible_actors=["Tutor académico", "Compañeros tutores entre pares"],
            time_horizon="Contacto en menos de una semana; revisión quincenal.",
            rationale=(
                "Albán-Holguín et al. (2025) identifican que los estudiantes de "
                "riesgo medio responden particularmente bien a intervenciones "
                "basadas en pertenencia social, una de las dos dimensiones del "
                "modelo clásico de Tinto (1987) aplicado al contexto digital."
            ),
        ),

        "bajo": RiskProfile(
            level="bajo",
            probability_range=(0.0, 0.50),
            typical_signals=[
                "Actividad regular en el LMS con clics distribuidos en el tiempo.",
                "Calificaciones medias en el percentil 50 o superior.",
                "Entregas puntuales de las evaluaciones programadas.",
                "Participación habitual en foros y consulta de materiales.",
            ],
            recommended_actions=[
                "Mensaje motivacional periódico reforzando el progreso observado.",
                "Inclusión en programas de mentoría inversa: convertir al "
                "estudiante en mentor de compañeros con mayor riesgo.",
                "Oferta de retos académicos opcionales (gamificación) que "
                "consolide su patrón positivo de aprendizaje.",
                "Encuesta de satisfacción semestral para detectar señales "
                "tempranas de desmotivación antes de que se reflejen en métricas.",
            ],
            responsible_actors=["Tutor académico"],
            time_horizon="Comunicación mensual; revisión semestral del perfil.",
            rationale=(
                "Núñez Collantes et al. (2026) demuestran que las intervenciones "
                "que combinan refuerzo positivo con gamificación elevan los "
                "indicadores de compromiso entre un 70 % y un 84 % en estudiantes "
                "que ya muestran trayectorias estables."
            ),
        ),
    }


# ============================================================================
# 3. CLASIFICACIÓN DE ESTUDIANTES SEGÚN PERFIL DE RIESGO
# ============================================================================

def classify_students_by_risk(
    y_proba: np.ndarray,
    profiles: dict[str, RiskProfile],
) -> pd.Series:
    """
    Asigna cada estudiante a un perfil de riesgo a partir de su probabilidad
    predicha por el mejor modelo.
    """
    labels = np.empty(len(y_proba), dtype=object)
    for name, profile in profiles.items():
        lo, hi = profile.probability_range
        mask = (y_proba >= lo) & (y_proba < hi if hi < 1.0 else y_proba <= hi)
        labels[mask] = name
    return pd.Series(labels, name="risk_level")


# ============================================================================
# 4. PROTOCOLO DE VALIDACIÓN CON TUTORES (ESCALA LIKERT)
# ============================================================================

@dataclass
class LikertScale:
    """Define la escala Likert de seis dimensiones del protocolo de validación."""
    dimensions: tuple = (
        "comprensibilidad",
        "aplicabilidad",
        "pertinencia",
        "novedad",
        "factibilidad",
        "impacto_percibido",
    )
    levels: dict = field(default_factory=lambda: {
        1: "Muy en desacuerdo / Muy bajo",
        2: "En desacuerdo / Bajo",
        3: "Neutral / Medio",
        4: "De acuerdo / Alto",
        5: "Muy de acuerdo / Muy alto",
    })


def build_validation_questionnaire(
    profiles: dict[str, RiskProfile],
) -> pd.DataFrame:
    """
    Genera el cuestionario que se entregará a cada tutor experto.

    El cuestionario incluye una fila por (perfil_riesgo, dimensión_Likert),
    con espacio para la puntuación numérica (1–5) y un campo de observación
    cualitativa. El resultado se exporta como CSV para ser distribuido y
    posteriormente reagregado.
    """
    scale = LikertScale()
    rows = []
    for level, profile in profiles.items():
        for dim in scale.dimensions:
            rows.append({
                "perfil_riesgo": level,
                "dimension_Likert": dim,
                "descripcion_perfil": "; ".join(profile.typical_signals[:3]),
                "acciones_propuestas": " | ".join(profile.recommended_actions[:3]),
                "puntuacion_1_a_5": "",
                "observacion_cualitativa": "",
            })
    df = pd.DataFrame(rows)
    return df


def aggregate_validation_responses(
    responses: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Reagrega las respuestas de los tutores en una tabla final con la media,
    la mediana y el coeficiente de variación por perfil y dimensión.

    Un coeficiente de variación bajo (< 0.20) indica acuerdo entre evaluadores;
    un valor alto identifica las dimensiones que requieren refinamiento del
    catálogo de recomendaciones.
    """
    if not responses:
        raise ValueError("La lista de respuestas no puede estar vacía.")

    long = pd.concat(
        [r.assign(evaluador=f"tutor_{i + 1}") for i, r in enumerate(responses)],
        ignore_index=True,
    )
    long["puntuacion_1_a_5"] = pd.to_numeric(long["puntuacion_1_a_5"], errors="coerce")

    summary = (
        long.groupby(["perfil_riesgo", "dimension_Likert"], observed=True)
            .agg(
                media=("puntuacion_1_a_5", "mean"),
                mediana=("puntuacion_1_a_5", "median"),
                desviacion=("puntuacion_1_a_5", "std"),
                n_evaluadores=("puntuacion_1_a_5", "count"),
            )
            .reset_index()
    )
    summary["coef_variacion"] = (
        summary["desviacion"] / summary["media"].replace(0, np.nan)
    ).fillna(0).round(3)
    return summary


if __name__ == "__main__":
    print("Módulo F5 (validación pedagógica) cargado correctamente.")
    profiles = get_baseline_risk_profiles()
    questionnaire = build_validation_questionnaire(profiles)
    print(questionnaire.head())
