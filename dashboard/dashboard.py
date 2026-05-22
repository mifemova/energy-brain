"""
================================================================================
TFM - Aplicación de la IA en la Innovación Educativa
Universidad Internacional de La Rioja (UNIR)
================================================================================

dashboard.py — Dashboard interactivo en Streamlit

Ejecución:
    streamlit run dashboard/dashboard.py -- --results-dir ../results

El dashboard consume el archivo dashboard_data.json que produce el pipeline
principal y ofrece cinco pestañas que reflejan las cinco fases del piloto:

    1. Resumen del experimento (balance de clases, distribución de riesgo).
    2. Comparativa de modelos (métricas globales y por pliegue).
    3. Explicabilidad SHAP (importancia global por modelo, imágenes).
    4. Consistencia estadística (Kruskal-Wallis, predictores robustos).
    5. Recomendaciones pedagógicas (catálogo por perfil de riesgo).

Diseñado para ser ejecutado en local durante la defensa del TFM, mostrando
los resultados en tiempo real desde el notebook que el director vea ejecutar.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE LA PÁGINA
# ────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TFM — Predicción de Abandono OULAD",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta institucional (azul UNIR + acentos pedagógicos)
COLORS = {
    "primary":   "#1A5490",
    "secondary": "#4A90D9",
    "accent":    "#F39C12",
    "danger":    "#C0392B",
    "warning":   "#E67E22",
    "success":   "#27AE60",
    "neutral":   "#7F8C8D",
}


# ────────────────────────────────────────────────────────────────────────
# CARGA DE LOS RESULTADOS DEL PIPELINE
# ────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_dashboard_data(results_dir: str) -> dict:
    """Carga dashboard_data.json y artefactos auxiliares del pipeline."""
    path = Path(results_dir) / "dashboard_data.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_results_dir() -> str:
    """Permite especificar el directorio mediante variable de entorno o CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=os.environ.get(
        "RESULTS_DIR", "./results"))
    args, _ = parser.parse_known_args()
    return args.results_dir


# ────────────────────────────────────────────────────────────────────────
# COMPONENTES VISUALES
# ────────────────────────────────────────────────────────────────────────

def render_header():
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {COLORS['primary']} 0%,
                    {COLORS['secondary']} 100%);
                    padding: 1.5rem; border-radius: 8px; color: white;
                    margin-bottom: 1.5rem;">
            <h1 style="margin: 0; color: white;">
                Predicción del Abandono en Educación Virtual
            </h1>
            <p style="margin: 0.25rem 0 0 0; opacity: 0.92;">
                Random Forest · XGBoost · CatBoost · SHAP — Open University
                Learning Analytics Dataset
            </p>
            <p style="margin: 0.25rem 0 0 0; opacity: 0.78; font-size: 0.85rem;">
                Trabajo Fin de Máster — Universidad Internacional de La Rioja
                (UNIR), 2026
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(results_dir: str):
    st.warning(
        f"No se encontró `dashboard_data.json` en `{results_dir}`. "
        "Ejecuta primero el pipeline principal:\n\n"
        "```bash\npython main_pipeline.py --data-path ./data/oulad --output-dir ./results\n```",
    )
    st.info("El dashboard se cargará automáticamente cuando los resultados "
            "estén disponibles.")


# ── TAB 1: Resumen ──────────────────────────────────────────────────────

def render_summary_tab(data: dict):
    st.subheader("Resumen del experimento")

    cb = data.get("class_balance", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Estudiantes totales", f"{cb.get('n_total', 0):,}")
    col2.metric("Abandonos (Withdrawn)", f"{cb.get('n_withdrawn', 0):,}")
    col3.metric("Permanencias", f"{cb.get('n_not_withdrawn', 0):,}")
    col4.metric("% Abandono", f"{cb.get('withdrawn_pct', 0):.2f} %")

    # Distribución de riesgo del mejor modelo
    risk_dist = data.get("risk_distribution", {})
    if risk_dist:
        st.markdown("#### Distribución del riesgo predicho")
        df_risk = pd.DataFrame([
            {"nivel": k, "proporcion": v} for k, v in risk_dist.items()
        ])
        color_map = {
            "alto": COLORS["danger"],
            "medio": COLORS["warning"],
            "bajo": COLORS["success"],
        }
        fig = px.bar(
            df_risk, x="nivel", y="proporcion",
            color="nivel", color_discrete_map=color_map,
            text=df_risk["proporcion"].apply(lambda v: f"{100*v:.1f} %"),
            labels={"nivel": "Nivel de riesgo",
                    "proporcion": "Proporción de estudiantes"},
            category_orders={"nivel": ["alto", "medio", "bajo"]},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=350, margin=dict(t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)


# ── TAB 2: Comparativa de modelos ───────────────────────────────────────

def render_models_tab(data: dict):
    st.subheader("Comparativa de modelos")

    comp = data.get("comparison_table", [])
    if not comp:
        st.info("Sin datos de comparación disponibles.")
        return

    df = pd.DataFrame(comp)
    st.markdown("#### Tabla comparativa global")
    st.dataframe(
        df.style.background_gradient(
            subset=["AUC-ROC (media)", "Recall Withdrawn", "F1 ponderado"],
            cmap="Blues",
        ),
        use_container_width=True, hide_index=True,
    )

    # Gráfico de barras con las tres métricas clave
    metrics_long = df.melt(
        id_vars=["Modelo"],
        value_vars=["AUC-ROC (media)", "Recall Withdrawn", "F1 ponderado"],
        var_name="Métrica", value_name="Valor",
    )
    fig = px.bar(
        metrics_long, x="Modelo", y="Valor", color="Métrica",
        barmode="group", text=metrics_long["Valor"].apply(lambda v: f"{v:.3f}"),
        color_discrete_sequence=[COLORS["primary"], COLORS["accent"],
                                 COLORS["success"]],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=420, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    # Distribución por pliegue
    st.markdown("#### Variabilidad por pliegue (validación cruzada)")
    per_fold = data.get("per_fold_metrics", {})
    if per_fold:
        rows = []
        for model, fold_list in per_fold.items():
            for r in fold_list:
                rows.append({"Modelo": model, **r})
        long_df = pd.DataFrame(rows)
        fig_box = px.box(
            long_df, x="Modelo", y="auc_roc",
            color="Modelo",
            color_discrete_sequence=[COLORS["primary"], COLORS["secondary"],
                                     COLORS["accent"]],
            points="all", labels={"auc_roc": "AUC-ROC"},
        )
        fig_box.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)


# ── TAB 3: SHAP ─────────────────────────────────────────────────────────

def render_shap_tab(data: dict, results_dir: str):
    st.subheader("Explicabilidad SHAP")

    importance = data.get("global_shap_importance", {})
    if not importance:
        st.info("Sin datos SHAP disponibles.")
        return

    model_choice = st.selectbox(
        "Modelo a inspeccionar", options=list(importance.keys()),
    )
    items = importance[model_choice]
    df_imp = pd.DataFrame([
        {"variable": k, "importancia": v} for k, v in items.items()
    ]).sort_values("importancia", ascending=True)

    fig = px.bar(
        df_imp, x="importancia", y="variable", orientation="h",
        color="importancia", color_continuous_scale="Blues",
        labels={"importancia": "Media de |SHAP|", "variable": "Variable"},
    )
    fig.update_layout(height=500, coloraxis_showscale=False,
                      margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    # Imágenes de los plots SHAP generados por F3
    st.markdown("#### Visualizaciones SHAP generadas")
    img_types = [
        ("summary",         "Summary plot (importancia global)"),
        ("dependence_top1", "Dependence plot - variable más importante"),
        ("waterfall_high",  "Waterfall - perfil de riesgo alto"),
        ("waterfall_medium", "Waterfall - perfil de riesgo medio"),
        ("waterfall_low",   "Waterfall - perfil de riesgo bajo"),
    ]
    for img_key, caption in img_types:
        path = Path(results_dir) / f"shap_{img_key}_{model_choice}.png"
        if path.exists():
            st.image(str(path), caption=caption)


# ── TAB 4: Consistencia estadística ─────────────────────────────────────

def render_consistency_tab(data: dict):
    st.subheader("Consistencia estadística entre modelos (OE4)")

    robust = data.get("robust_predictors", [])
    dependent = data.get("algorithm_dependent", [])

    col1, col2 = st.columns(2)
    col1.metric("Predictores robustos", len(robust),
                help="Sin diferencias significativas entre algoritmos")
    col2.metric("Dependientes del algoritmo", len(dependent),
                help="Con diferencias significativas Kruskal-Wallis + Bonferroni")

    st.markdown("#### Variables robustas (generalizables)")
    if robust:
        st.success("✅ " + " · ".join(f"`{v}`" for v in robust))
    else:
        st.info("Aún no se han identificado predictores robustos.")

    st.markdown("#### Variables algoritmo-dependientes")
    if dependent:
        st.warning("⚠️ " + " · ".join(f"`{v}`" for v in dependent))

    kw = data.get("kruskal_results", [])
    if kw:
        df_kw = pd.DataFrame(kw)
        if "mediana_SHAP_por_modelo" in df_kw.columns:
            df_kw = df_kw.drop(columns=["mediana_SHAP_por_modelo"])
        st.markdown("#### Resultados de la prueba Kruskal-Wallis")
        st.dataframe(df_kw.sort_values("p_value"),
                     use_container_width=True, hide_index=True)


# ── TAB 5: Recomendaciones pedagógicas ──────────────────────────────────

def render_pedagogy_tab(data: dict):
    st.subheader("Recomendaciones pedagógicas (OE5)")

    profiles = data.get("profiles", {})
    if not profiles:
        st.info("Sin recomendaciones disponibles.")
        return

    level_colors = {
        "alto": COLORS["danger"],
        "medio": COLORS["warning"],
        "bajo": COLORS["success"],
    }
    icons = {"alto": "🚨", "medio": "⚠️", "bajo": "✅"}

    for level, profile in profiles.items():
        color = level_colors.get(level, COLORS["neutral"])
        icon = icons.get(level, "·")
        pr = profile.get("probability_range", [0, 0])
        st.markdown(
            f"""
            <div style="border-left: 5px solid {color}; padding: 0.75rem 1rem;
                        background: rgba(0,0,0,0.02); border-radius: 4px;
                        margin: 0.6rem 0;">
                <h3 style="margin: 0; color: {color};">
                    {icon} Perfil de riesgo {level.upper()}
                </h3>
                <p style="margin: 0.25rem 0; font-size: 0.85rem; color: #666;">
                    Probabilidad predicha: [{pr[0]:.2f}, {pr[1]:.2f}] —
                    Horizonte: {profile.get('time_horizon', '')}
                </p>
            </div>
            """, unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Señales típicas detectadas (SHAP):**")
            for signal in profile.get("typical_signals", []):
                st.markdown(f"• {signal}")
        with col2:
            st.markdown("**Acciones recomendadas:**")
            for action in profile.get("recommended_actions", []):
                st.markdown(f"• {action}")

        actors = profile.get("responsible_actors", [])
        if actors:
            st.markdown(
                "**Responsables institucionales:** " +
                " · ".join(f"`{a}`" for a in actors)
            )
        st.divider()


# ────────────────────────────────────────────────────────────────────────
# APLICACIÓN PRINCIPAL
# ────────────────────────────────────────────────────────────────────────

def main():
    results_dir = get_results_dir()
    render_header()

    data = load_dashboard_data(results_dir)
    if not data:
        render_empty_state(results_dir)
        return

    with st.sidebar:
        st.markdown("## Navegación")
        st.markdown(f"**Directorio:** `{results_dir}`")
        st.markdown("---")
        st.markdown("### Contexto del piloto")
        st.markdown(
            "Este dashboard visualiza los resultados del piloto experimental "
            "del TFM grupal, ejecutado sobre el Open University Learning "
            "Analytics Dataset (OULAD)."
        )
        st.markdown("---")
        st.markdown("### Fases del pipeline")
        st.markdown(
            "- **F1**: Preprocesamiento e ingeniería\n"
            "- **F2**: Entrenamiento y CV anidado\n"
            "- **F3**: Análisis SHAP\n"
            "- **F4**: Consistencia estadística\n"
            "- **F5**: Validación pedagógica"
        )

    tabs = st.tabs([
        "📊 Resumen", "🤖 Modelos", "🔍 SHAP",
        "📈 Consistencia", "🎓 Pedagogía",
    ])
    with tabs[0]: render_summary_tab(data)
    with tabs[1]: render_models_tab(data)
    with tabs[2]: render_shap_tab(data, results_dir)
    with tabs[3]: render_consistency_tab(data)
    with tabs[4]: render_pedagogy_tab(data)


if __name__ == "__main__":
    main()
