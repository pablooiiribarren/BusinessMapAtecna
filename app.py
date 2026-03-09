from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

import src.manifest as mf
from src.data_prep import load_businessmap_workbook, prepare_from_frames
from src.pipeline import run_baseline_pipeline
from src.type_segmentation import run_type_segmented_analysis

# ── constants ─────────────────────────────────────────────────────────────────
FORECAST_COLOR = {
    "no_throughput": "#7B2D8B",
    "overload":      "#C8553D",
    "risk":          "#E8A838",
    "healthy":       "#4CAF50",
}
ALERT_COLOR = {
    "bottleneck": "#C8553D",
    "risk":       "#E8A838",
    "healthy":    "#4CAF50",
}
BOTTLENECK_COLOR = {
    "bottleneck": "#C8553D",
    "risk":       "#E8A838",
    "healthy":    "#4CAF50",
}

TYPE_OPTIONS = ["Todos", "PROYECTO EXTERNO", "PROYECTO INTERNO", "PRODUCTO"]

COLUMN_LABELS: dict[str, str] = {
    # Identificadores
    "Owner":                      "Responsable",
    "Type Name":                  "Tipo de tarea",
    "Column Name":                "Fase actual",
    "Card ID":                    "ID Tarea",
    # Forecast
    "current_wip":                "Tareas en curso ahora",
    "forecast_wip":               "Previsión de tareas",
    "forecast_wip_low":           "Previsión mínima",
    "forecast_wip_high":          "Previsión máxima",
    "arrival_rate_per_day":       "Entradas/día (media)",
    "completion_rate_per_day":    "Cierres/día (media)",
    "expected_arrivals":          "Entradas previstas",
    "expected_completions":       "Cierres previstos",
    "status":                     "Estado",
    "status_reason":              "Motivo del estado",
    # Alertas por tarea
    "age_days":                   "Días abierta",
    "age_vs_benchmark":           "Veces el tiempo habitual",
    "days_since_last_moved":      "Días sin avance",
    "benchmark_median_days":      "Tiempo habitual (mediana)",
    "benchmark_p90_days":         "Tiempo habitual (P90)",
    "alert_score":                "Puntuación de riesgo",
    "alert_level":                "Nivel de alerta",
    "alert_reason":               "Causas de la alerta",
    # Bottlenecks por responsable
    "open_tasks":                 "Tareas abiertas",
    "bottleneck_tasks":           "Tareas críticas",
    "risk_tasks":                 "Tareas en riesgo",
    "currently_blocked_tasks":    "Tareas bloqueadas",
    "stagnant_tasks":             "Tareas estancadas",
    "old_open_tasks":             "Tareas antiguas",
    "dependency_risk_tasks":      "Riesgo por dependencias",
    "median_open_age_days":       "Antigüedad mediana (días)",
    "max_open_age_days":          "Antigüedad máxima (días)",
    "dominant_column":            "Fase con más carga",
    "forecast_status":            "Estado de capacidad",
    "bottleneck_score":           "Puntuación de cuello de botella",
    "bottleneck_status":          "Situación",
    # Bottlenecks por columna
    "column_name":                "Fase Kanban",
    "owners_involved":            "Responsables implicados",
    "median_age_days":            "Antigüedad mediana (días)",
    "max_age_days":               "Antigüedad máxima (días)",
    "mean_days_since_last_moved": "Media de días sin avance",
}

# Traducción de valores de estado (inglés → español)
STATUS_LABELS: dict[str, str] = {
    "no_throughput": "Sin actividad reciente",
    "overload":      "Sobrecarga",
    "risk":          "En riesgo",
    "healthy":       "Capacidad OK",
    "bottleneck":    "Cuello de botella",
}

# Mapas de color con claves en español (para tablas con valores ya traducidos)
FORECAST_COLOR_ES    = {STATUS_LABELS.get(k, k): v for k, v in FORECAST_COLOR.items()}
ALERT_COLOR_ES       = {STATUS_LABELS.get(k, k): v for k, v in ALERT_COLOR.items()}
BOTTLENECK_COLOR_ES  = {STATUS_LABELS.get(k, k): v for k, v in BOTTLENECK_COLOR.items()}

# Traducción de tokens de alert_reason
ALERT_REASON_LABELS: dict[str, str] = {
    "old_open":             "Tarea antigua",
    "stagnant":             "Sin avance reciente",
    "older_than_history":   "Supera el benchmark histórico",
    "dependency_risk":      "Riesgo por dependencias",
    "complexity_risk":      "Alta complejidad",
    "currently_blocked":    "Bloqueada actualmente",
    "historical_block":     "Historial de bloqueos",
    "healthy":              "Sin alertas",
}

# Traducción de tokens de status_reason (forecast)
STATUS_REASON_LABELS: dict[str, str] = {
    "arrivals > completions":              "Entradas superan cierres",
    "projected backlog exceeds horizon":   "Cartera prevista supera el horizonte",
    "capacity aligned with horizon":       "Capacidad alineada con el horizonte",
    "open work but no recent completions": "Trabajo pendiente sin cierres recientes",
    "no recent load":                      "Sin carga reciente",
}


def _translate_status(series: pd.Series) -> pd.Series:
    return series.map(lambda v: STATUS_LABELS.get(v, v) if pd.notna(v) else v)


def _translate_alert_reason(series: pd.Series) -> pd.Series:
    def _tok(reason: str) -> str:
        if not isinstance(reason, str):
            return reason
        parts = []
        for token in reason.split("; "):
            token = token.strip()
            if token.startswith("owner_"):
                suffix = token[len("owner_"):]
                parts.append(f"Responsable {STATUS_LABELS.get(suffix, suffix)}")
            else:
                parts.append(ALERT_REASON_LABELS.get(token, token))
        return "; ".join(parts)
    return series.map(lambda v: _tok(v) if pd.notna(v) else v)


def _translate_status_reason(series: pd.Series) -> pd.Series:
    def _tok(reason: str) -> str:
        if not isinstance(reason, str):
            return reason
        parts = [STATUS_REASON_LABELS.get(t.strip(), t.strip()) for t in reason.split("; ")]
        return "; ".join(parts)
    return series.map(lambda v: _tok(v) if pd.notna(v) else v)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BusinessMap Analytics",
    page_icon="📊",
    layout="wide",
)

# ── session state init ────────────────────────────────────────────────────────
if "manifest" not in st.session_state:
    st.session_state.manifest = mf.load_manifest()
if "trained_paths" not in st.session_state:
    st.session_state.trained_paths = mf.active_paths(st.session_state.manifest)

manifest: dict = st.session_state.manifest

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuración")

    rate_window = st.slider(
        "Ventana de tasas (días)",
        min_value=15, max_value=120, value=60, step=5,
        help="Días para calcular tasas de llegada y completado.",
    )
    horizon = st.slider(
        "Horizonte de forecast (días)",
        min_value=3, max_value=30, value=5, step=1,
        help="Días hacia adelante para el forecast de WIP.",
    )

    st.divider()

    selected_type = st.radio("Segmento a analizar", TYPE_OPTIONS, index=0)

    st.divider()

    # Estado del dataset en sidebar
    n_active = sum(
        1 for f in manifest["files"]
        if f["active"] and not f.get("missing")
    )
    current_paths = mf.active_paths(manifest)
    needs_retrain = current_paths != st.session_state.trained_paths

    if needs_retrain:
        st.warning("⚠️ Dataset modificado.\nVe a **Gestión de datos** y pulsa **Reentrenar**.")
    else:
        st.success(f"✅ {n_active} archivo{'s' if n_active != 1 else ''} activo{'s' if n_active != 1 else ''}")

# ── pipeline (cached por trained_paths + parámetros) ─────────────────────────
@st.cache_data(show_spinner="Ejecutando pipeline de análisis…")
def compute_artifacts(
    trained_paths: tuple[str, ...],
    rate_window: int,
    horizon: int,
) -> tuple[dict, dict]:
    """Carga y combina los archivos activos, ejecuta el pipeline completo."""
    if not trained_paths:
        raise ValueError("No hay archivos activos en el dataset.")

    all_bm, all_links, all_subtasks = [], [], []
    for path in trained_paths:
        bm, links, subtasks = load_businessmap_workbook(path)
        all_bm.append(bm)
        all_links.append(links)
        all_subtasks.append(subtasks)

    if len(all_bm) == 1:
        merged_bm, merged_links, merged_subtasks = all_bm[0], all_links[0], all_subtasks[0]
    else:
        merged_bm = (
            pd.concat(all_bm, ignore_index=True)
            .drop_duplicates(subset=["Card ID"], keep="last")
            .reset_index(drop=True)
        )
        merged_links = (
            pd.concat(all_links, ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )
        merged_subtasks = (
            pd.concat(all_subtasks, ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )

    prepared = prepare_from_frames(merged_bm, merged_links, merged_subtasks)
    base = run_baseline_pipeline(
        prepared=prepared,
        rate_window_days=rate_window,
        dashboard_horizon_days=horizon,
        scenario_horizons=(5, 10, 20),
    )
    segmented = run_type_segmented_analysis(
        base["bm"],
        reference_date=base["reference_date"],
        rate_window_days=rate_window,
        dashboard_horizon_days=horizon,
        scenario_horizons=(5, 10, 20),
    )
    return base, segmented


try:
    base_artifacts, segmented = compute_artifacts(
        st.session_state.trained_paths, rate_window, horizon
    )
except Exception as exc:
    st.error(f"❌ Error al procesar el dataset: {exc}")
    if needs_retrain:
        st.info("Puede que un archivo activo haya desaparecido del disco. "
                "Ve a **Gestión de datos**, revisa la lista y pulsa **Reentrenar**.")
    st.stop()

# ── resolve active artifacts by segment ───────────────────────────────────────
if selected_type == "Todos":
    forecast_df = base_artifacts["forecast_dashboard_export"]
    task_alerts = base_artifacts["task_alerts"]
    owner_bns   = base_artifacts["owner_bottlenecks"]
    column_bns  = base_artifacts["column_bottlenecks"]
    scenarios   = base_artifacts["forecast_scenarios"]
else:
    art         = segmented["type_artifacts"].get(selected_type, {})
    forecast_df = art.get("forecast_dashboard_export", pd.DataFrame())
    task_alerts = art.get("task_alerts",               pd.DataFrame())
    owner_bns   = art.get("owner_bottlenecks",         pd.DataFrame())
    column_bns  = art.get("column_bottlenecks",        pd.DataFrame())
    scenarios   = art.get("forecast_scenarios",         pd.DataFrame())

ref_date  = base_artifacts["reference_date"].strftime("%d/%m/%Y")
n_trained = len(st.session_state.trained_paths)
n_total   = len(base_artifacts["bm"])
n_open    = int(base_artifacts["bm"]["Actual End Date"].isna().sum())
n_closed  = n_total - n_open

# ── header ────────────────────────────────────────────────────────────────────
st.title("📊 BusinessMap Analytics")
st.caption(
    f"Fecha de referencia: **{ref_date}** · "
    f"Ventana: **{rate_window} días** · "
    f"Horizonte: **{horizon} días** · "
    f"Segmento: **{selected_type}** · "
    f"Archivos en uso: **{n_trained}** · "
    f"Total tarjetas: **{n_total}** ({n_open} abiertas / {n_closed} cerradas)"
)

# ── helpers ───────────────────────────────────────────────────────────────────
def _lbl(col: str) -> str:
    return COLUMN_LABELS.get(col, col)


def _color_col(series: pd.Series, color_map: dict) -> list[str]:
    return [
        f"background-color: {color_map.get(v, '#ffffff')}22; "
        f"color: {color_map.get(v, '#333333')}; font-weight: 600"
        for v in series
    ]


def _prepare(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    available = [c for c in cols if c in df.columns]
    return df[available].rename(columns=COLUMN_LABELS)


def _safe_style(df: pd.DataFrame, col: str, color_map: dict) -> pd.io.formats.style.Styler:
    styler = df.style
    if col in df.columns:
        styler = styler.apply(lambda s: _color_col(s, color_map), subset=[col])
    return styler


# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Predicción de carga",
    "⏱️ Tiempo real por tarea",
    "🚧 Cuellos de botella",
    "🗂️ Gestión de datos",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICCIÓN DE CARGA FUTURA
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    if forecast_df.empty:
        st.info("No hay datos de forecast para el segmento seleccionado.")
    else:
        total_cur  = int(forecast_df["current_wip"].sum())
        total_fore = round(float(forecast_df["forecast_wip"].sum()), 1)
        n_notp     = int((forecast_df["status"] == "no_throughput").sum())
        n_over     = int((forecast_df["status"] == "overload").sum())
        n_risk     = int((forecast_df["status"] == "risk").sum())
        n_healthy  = int((forecast_df["status"] == "healthy").sum())

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("WIP actual",                 total_cur)
        c2.metric(f"WIP forecast ({horizon}d)", total_fore,
                  delta=round(total_fore - total_cur, 1))
        c3.metric("🟣 Sin throughput",          n_notp)
        c4.metric("🟠 En riesgo / sobrecarga",  n_risk + n_over)
        c5.metric("🟢 Saludables",              n_healthy)

        st.divider()
        col_chart, col_table = st.columns(2)

        with col_chart:
            st.subheader("Previsión de tareas en curso por responsable")
            chart_df = forecast_df.copy()
            chart_df["status_es"] = chart_df["status"].map(lambda v: STATUS_LABELS.get(v, v))
            fig = px.bar(
                chart_df.sort_values("forecast_wip", ascending=True),
                x="forecast_wip", y="Owner",
                color="status_es", color_discrete_map=FORECAST_COLOR_ES,
                orientation="h",
                labels={
                    "forecast_wip": f"Previsión de tareas ({horizon} días)",
                    "Owner":        "Responsable",
                    "status_es":    "Estado",
                },
                height=max(350, len(forecast_df) * 30),
            )
            fig.update_layout(margin=dict(l=0, r=10, t=10, b=0), legend_title_text="Estado")
            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            st.subheader("Detalle por responsable")
            disp = _prepare(forecast_df, [
                "Owner", "current_wip", "forecast_wip",
                "forecast_wip_low", "forecast_wip_high",
                "arrival_rate_per_day", "completion_rate_per_day",
                "expected_arrivals", "expected_completions",
                "status", "status_reason",
            ])
            if _lbl("status") in disp.columns:
                disp[_lbl("status")] = _translate_status(disp[_lbl("status")])
            if _lbl("status_reason") in disp.columns:
                disp[_lbl("status_reason")] = _translate_status_reason(disp[_lbl("status_reason")])
            styled = _safe_style(disp, _lbl("status"), FORECAST_COLOR_ES).format(
                {
                    _lbl("arrival_rate_per_day"):    "{:.3f}",
                    _lbl("completion_rate_per_day"): "{:.3f}",
                    _lbl("forecast_wip"):            "{:.1f}",
                    _lbl("forecast_wip_low"):        "{:.1f}",
                    _lbl("forecast_wip_high"):       "{:.1f}",
                    _lbl("current_wip"):             "{:.0f}",
                    _lbl("expected_arrivals"):        "{:.1f}",
                    _lbl("expected_completions"):     "{:.1f}",
                },
                na_rep="-",
            )
            st.dataframe(styled, use_container_width=True,
                         height=max(350, len(forecast_df) * 35 + 38))

        if scenarios is not None and not (
            isinstance(scenarios, pd.DataFrame) and scenarios.empty
        ):
            st.divider()
            st.subheader("Escenarios por horizonte (5 / 10 / 20 días)")
            scen_rename = {"Owner": "Responsable"}
            for h in (5, 10, 20):
                scen_rename[f"forecast_wip_{h}d"] = f"Previsión {h} días"
                scen_rename[f"status_{h}d"]        = f"Estado {h} días"
            scen_disp = scenarios.rename(columns=scen_rename)
            for col in scen_disp.columns:
                if col.startswith("Estado"):
                    scen_disp[col] = _translate_status(scen_disp[col])
            st.dataframe(scen_disp, use_container_width=True)

        if selected_type == "Todos":
            st.divider()
            st.subheader("Comparativa entre tipos de tarea")

            # Totales globales
            type_summary = segmented.get("type_summary", pd.DataFrame())
            if not type_summary.empty:
                t1, t2, t3, t4 = st.columns(4)
                t1.metric("Total tarjetas",  n_total)
                t2.metric("Abiertas",        n_open)
                t3.metric("Cerradas",        n_closed)
                t4.metric("Archivos en uso", n_trained)

                st.dataframe(
                    type_summary.rename(columns={
                        "type_name":                   "Tipo",
                        "n_rows":                      "Total",
                        "n_open":                      "Abiertas",
                        "n_closed":                    "Cerradas",
                        "owners":                      "Propietarios",
                        "median_closed_duration_days": "Duración mediana (días)",
                        "p90_closed_duration_days":    "Duración P90 (días)",
                        "median_open_age_days":        "Antigüedad mediana abierta",
                        "max_open_age_days":           "Antigüedad máxima abierta",
                    }),
                    use_container_width=True,
                )
                st.divider()

            type_fore = segmented.get("type_forecast_overview", pd.DataFrame())
            if not type_fore.empty:
                col_a, col_b = st.columns(2)
                with col_a:
                    fig_t = px.bar(
                        type_fore, x="type_name", y="forecast_wip",
                        color="type_name",
                        labels={
                            "type_name":   "Tipo de tarea",
                            "forecast_wip": f"Previsión de tareas ({horizon} días)",
                        },
                        title=f"Previsión de tareas por tipo ({horizon} días)",
                    )
                    fig_t.update_layout(showlegend=False, margin=dict(t=30, b=0))
                    st.plotly_chart(fig_t, use_container_width=True)
                with col_b:
                    st.dataframe(
                        type_fore.rename(columns={
                            "type_name":            "Tipo de tarea",
                            "owners_in_forecast":   "Responsables",
                            "current_wip":          "Tareas en curso ahora",
                            "forecast_wip":         "Previsión de tareas",
                            "no_throughput_owners": "Sin actividad reciente",
                            "overload_owners":      "Sobrecarga",
                            "risk_owners":          "En riesgo",
                            "healthy_owners":       "Capacidad OK",
                        }),
                        use_container_width=True,
                    )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — ESTIMACIÓN DE TIEMPO REAL POR TAREA
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    if task_alerts.empty:
        st.info("No hay tareas abiertas para el segmento seleccionado.")
    else:
        n_bt = int((task_alerts["alert_level"] == "bottleneck").sum())
        n_rk = int((task_alerts["alert_level"] == "risk").sum())
        n_ok = int((task_alerts["alert_level"] == "healthy").sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tareas abiertas", len(task_alerts))
        c2.metric("🔴 Críticas",     n_bt)
        c3.metric("🟠 En riesgo",    n_rk)
        c4.metric("🟢 Sin alertas",  n_ok)

        st.divider()

        _ALERT_OPTIONS_ES = {
            "Tareas críticas": "bottleneck",
            "En riesgo":       "risk",
            "Sin alertas":     "healthy",
        }
        filter_level_es = st.multiselect(
            "Filtrar por nivel de alerta",
            options=list(_ALERT_OPTIONS_ES.keys()),
            default=["Tareas críticas", "En riesgo"],
        )
        filter_level = [_ALERT_OPTIONS_ES[k] for k in filter_level_es]
        filtered = (
            task_alerts[task_alerts["alert_level"].isin(filter_level)]
            if filter_level else task_alerts
        )

        col_charts, col_table = st.columns([1, 1.5])

        with col_charts:
            st.subheader("Antigüedad vs tiempo habitual histórico")
            chart_filtered = filtered.copy()
            chart_filtered["nivel_es"] = chart_filtered["alert_level"].map(
                lambda v: STATUS_LABELS.get(v, v)
            )
            fig2 = px.histogram(
                chart_filtered, x="age_vs_benchmark",
                color="nivel_es", color_discrete_map=ALERT_COLOR_ES,
                nbins=20,
                labels={
                    "age_vs_benchmark": "Veces el tiempo habitual",
                    "count":            "Tareas",
                    "nivel_es":         "Nivel de alerta",
                },
            )
            fig2.add_vline(x=1, line_dash="dash", line_color="gray",
                           annotation_text="= tiempo habitual")
            fig2.add_vline(x=3, line_dash="dot",  line_color="#C8553D",
                           annotation_text="3× tiempo habitual")
            fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Distribución por fase Kanban")
            col_dist = (
                chart_filtered.groupby(["Column Name", "nivel_es"])
                .size().reset_index(name="n")
            )
            fig3 = px.bar(
                col_dist, x="Column Name", y="n",
                color="nivel_es", color_discrete_map=ALERT_COLOR_ES,
                labels={"n": "Tareas", "Column Name": "Fase", "nivel_es": "Nivel de alerta"},
            )
            fig3.update_layout(xaxis_tickangle=-30, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig3, use_container_width=True)

        with col_table:
            st.subheader("Alertas por tarea")
            disp2 = _prepare(filtered, [
                "Card ID", "Owner", "Column Name", "Type Name",
                "age_days", "age_vs_benchmark", "days_since_last_moved",
                "benchmark_median_days", "benchmark_p90_days",
                "alert_score", "alert_level", "alert_reason",
            ])
            if _lbl("alert_level") in disp2.columns:
                disp2[_lbl("alert_level")] = _translate_status(disp2[_lbl("alert_level")])
            if _lbl("alert_reason") in disp2.columns:
                disp2[_lbl("alert_reason")] = _translate_alert_reason(disp2[_lbl("alert_reason")])
            styled2 = _safe_style(disp2, _lbl("alert_level"), ALERT_COLOR_ES).format(
                {
                    _lbl("age_days"):              "{:.1f}",
                    _lbl("age_vs_benchmark"):      "{:.2f}x",
                    _lbl("days_since_last_moved"): "{:.1f}",
                    _lbl("benchmark_median_days"): "{:.1f}",
                    _lbl("benchmark_p90_days"):    "{:.1f}",
                    _lbl("alert_score"):           "{:.0f}",
                },
                na_rep="-",
            )
            st.dataframe(styled2, use_container_width=True, height=560)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — CUELLOS DE BOTELLA
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Por responsable")
        if owner_bns.empty:
            st.info("Sin datos.")
        else:
            n_bt_o = int((owner_bns["bottleneck_status"] == "bottleneck").sum())
            n_rk_o = int((owner_bns["bottleneck_status"] == "risk").sum())
            n_ok_o = int((owner_bns["bottleneck_status"] == "healthy").sum())

            m1, m2, m3 = st.columns(3)
            m1.metric("🔴 Crítico",   n_bt_o)
            m2.metric("🟠 En riesgo", n_rk_o)
            m3.metric("🟢 OK",        n_ok_o)

            bn_chart_df = owner_bns.copy()
            bn_chart_df["situacion_es"] = bn_chart_df["bottleneck_status"].map(lambda v: STATUS_LABELS.get(v, v))
            fig4 = px.bar(
                bn_chart_df.sort_values("bottleneck_score", ascending=True),
                x="bottleneck_score", y="Owner",
                color="situacion_es", color_discrete_map=BOTTLENECK_COLOR_ES,
                orientation="h",
                labels={
                    "bottleneck_score": "Puntuación de cuello de botella",
                    "Owner":            "Responsable",
                    "situacion_es":     "Situación",
                },
                height=max(300, len(owner_bns) * 30),
            )
            fig4.update_layout(margin=dict(l=0, r=10, t=10, b=0))
            st.plotly_chart(fig4, use_container_width=True)

            disp3 = _prepare(owner_bns, [
                "Owner", "open_tasks", "bottleneck_tasks", "risk_tasks",
                "currently_blocked_tasks", "stagnant_tasks",
                "old_open_tasks", "dependency_risk_tasks",
                "median_open_age_days", "max_open_age_days",
                "dominant_column", "forecast_status",
                "bottleneck_score", "bottleneck_status",
            ])
            for col in (_lbl("forecast_status"), _lbl("bottleneck_status")):
                if col in disp3.columns:
                    disp3[col] = _translate_status(disp3[col])
            styled3 = _safe_style(
                disp3, _lbl("bottleneck_status"), BOTTLENECK_COLOR_ES
            ).format(
                {
                    _lbl("median_open_age_days"): "{:.1f}",
                    _lbl("max_open_age_days"):    "{:.1f}",
                    _lbl("bottleneck_score"):     "{:.0f}",
                },
                na_rep="-",
            )
            st.dataframe(styled3, use_container_width=True)

    with col_right:
        st.subheader("Por columna Kanban")
        if column_bns.empty:
            st.info("Sin datos.")
        else:
            col_chart_df = column_bns.copy()
            col_chart_df["situacion_es"] = col_chart_df["bottleneck_status"].map(lambda v: STATUS_LABELS.get(v, v))
            fig5 = px.bar(
                col_chart_df.sort_values("bottleneck_tasks", ascending=True),
                x="bottleneck_tasks", y="column_name",
                color="situacion_es", color_discrete_map=BOTTLENECK_COLOR_ES,
                orientation="h",
                labels={
                    "bottleneck_tasks": "Tareas críticas",
                    "column_name":      "Fase",
                    "situacion_es":     "Situación",
                },
            )
            fig5.update_layout(margin=dict(l=0, r=10, t=10, b=0))
            st.plotly_chart(fig5, use_container_width=True)

            disp4 = _prepare(column_bns, [
                "column_name", "open_tasks", "bottleneck_tasks", "risk_tasks",
                "owners_involved", "median_age_days", "max_age_days",
                "mean_days_since_last_moved",
                "dependency_risk_tasks", "currently_blocked_tasks",
                "bottleneck_status",
            ])
            if _lbl("bottleneck_status") in disp4.columns:
                disp4[_lbl("bottleneck_status")] = _translate_status(disp4[_lbl("bottleneck_status")])
            styled4 = _safe_style(
                disp4, _lbl("bottleneck_status"), BOTTLENECK_COLOR_ES
            ).format(
                {
                    _lbl("median_age_days"):            "{:.1f}",
                    _lbl("max_age_days"):               "{:.1f}",
                    _lbl("mean_days_since_last_moved"): "{:.1f}",
                },
                na_rep="-",
            )
            st.dataframe(styled4, use_container_width=True)

    if selected_type == "Todos":
        st.divider()
        col_bt_sum, col_bt_chart = st.columns(2)

        with col_bt_sum:
            st.subheader("Resumen de bottlenecks por tipo")
            type_bt_ov = segmented.get("type_bottleneck_overview", pd.DataFrame())
            if not type_bt_ov.empty:
                st.dataframe(
                    type_bt_ov.rename(columns={
                        "type_name":          "Tipo",
                        "open_tasks":         "Abiertas",
                        "bottleneck_tasks":   "Bottleneck",
                        "risk_tasks":         "Riesgo",
                        "healthy_tasks":      "Saludables",
                        "bottleneck_owners":  "Propietarios BN",
                        "risk_owners":        "Propietarios riesgo",
                        "healthy_owners":     "Propietarios OK",
                        "top_problem_column": "Columna problema",
                    }),
                    use_container_width=True,
                )

        with col_bt_chart:
            st.subheader("Tareas bottleneck por tipo")
            type_bt_ov = segmented.get("type_bottleneck_overview", pd.DataFrame())
            if not type_bt_ov.empty:
                fig6 = px.bar(
                    type_bt_ov, x="type_name",
                    y=["bottleneck_tasks", "risk_tasks", "healthy_tasks"],
                    color_discrete_map={
                        "bottleneck_tasks": "#C8553D",
                        "risk_tasks":       "#E8A838",
                        "healthy_tasks":    "#4CAF50",
                    },
                    labels={"type_name": "Tipo", "value": "Tareas", "variable": "Nivel"},
                    barmode="stack",
                )
                fig6.update_layout(margin=dict(t=10, b=0))
                st.plotly_chart(fig6, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — GESTIÓN DE DATOS
# ═════════════════════════════════════════════════════════════════════════════
with tab4:

    # ── callbacks ─────────────────────────────────────────────────────────────
    def _on_toggle(file_id: str, key: str) -> None:
        new_val = st.session_state[key]
        st.session_state.manifest = mf.toggle_active(
            st.session_state.manifest, file_id, new_val
        )

    def _on_delete(file_id: str) -> None:
        st.session_state.manifest = mf.remove_file(
            st.session_state.manifest, file_id
        )

    # ── sección: archivos registrados ─────────────────────────────────────────
    st.subheader("📁 Archivos en el dataset")
    st.caption(
        "El archivo base siempre está activo. "
        "Los archivos adicionales se pueden activar/desactivar o eliminar."
    )

    files = st.session_state.manifest["files"]

    for f in files:
        is_base    = f.get("is_base", False)
        is_missing = f.get("missing", False)

        col_chk, col_info, col_del = st.columns([0.5, 5, 0.7])

        with col_chk:
            chk_key = f"chk_{f['id']}"
            st.checkbox(
                label="",
                value=f["active"],
                key=chk_key,
                on_change=_on_toggle,
                args=(f["id"], chk_key),
                disabled=is_base or is_missing,
                label_visibility="collapsed",
            )

        with col_info:
            upload_dt = f.get("upload_date", "")[:10]
            if is_base:
                badge = "🔒 Base"
            elif is_missing:
                badge = "❌ Faltante"
            elif f["active"]:
                badge = "✅ Activo"
            else:
                badge = "⏸️ Inactivo"

            st.markdown(
                f"**{f['original_name']}** &nbsp; `{badge}` &nbsp; "
                f"<span style='color:gray;font-size:0.85em'>{upload_dt}</span>",
                unsafe_allow_html=True,
            )

        with col_del:
            st.button(
                "🗑️",
                key=f"del_{f['id']}",
                on_click=_on_delete,
                args=(f["id"],),
                disabled=is_base,
                help="Eliminar este archivo del dataset",
            )

    st.divider()

    # ── sección: subir nuevo archivo ──────────────────────────────────────────
    st.subheader("⬆️ Añadir nuevos datos")
    st.caption(
        "Sube un export de BusinessMap con las hojas **Businessmap**, **Links** y **Subtasks**. "
        "El nuevo archivo se sumará al dataset actual."
    )

    uploaded = st.file_uploader(
        "Selecciona un archivo Excel de BusinessMap",
        type=["xlsx"],
        key="data_uploader",
    )

    if uploaded is not None:
        file_bytes = uploaded.read()

        with st.spinner("Validando esquema…"):
            errors = mf.validate_schema(file_bytes)

        if errors:
            st.error("❌ El archivo no es compatible:")
            for err in errors:
                st.markdown(f"- {err}")
        else:
            st.success("✅ Esquema válido")
            if st.button("➕ Añadir al dataset", type="primary"):
                updated, err_msg = mf.add_file(
                    st.session_state.manifest,
                    uploaded.name,
                    file_bytes,
                )
                if err_msg:
                    st.warning(err_msg)
                else:
                    st.session_state.manifest = updated
                    st.success(f"«{uploaded.name}» añadido correctamente.")
                    st.rerun()

    st.divider()

    # ── sección: reentrenamiento ──────────────────────────────────────────────
    st.subheader("🔄 Reentrenamiento del pipeline")

    current_paths = mf.active_paths(st.session_state.manifest)
    needs_retrain = current_paths != st.session_state.trained_paths

    n_current = len(current_paths)
    n_trained  = len(st.session_state.trained_paths)

    last_hash = st.session_state.manifest.get("last_trained_hash")

    if needs_retrain:
        st.warning(
            f"La selección de archivos ha cambiado "
            f"({n_trained} → {n_current} archivo{'s' if n_current != 1 else ''}). "
            "Pulsa **Reentrenar** para actualizar los resultados del dashboard."
        )
    else:
        st.info(
            f"✅ El pipeline está actualizado con **{n_current} archivo{'s' if n_current != 1 else ''}**."
        )

    if st.button(
        "🔄 Reentrenar pipeline",
        type="primary",
        disabled=not needs_retrain,
        help="Ejecuta el pipeline con todos los archivos activos seleccionados.",
    ):
        with st.spinner("Reentrenando pipeline con el dataset combinado…"):
            st.session_state.trained_paths = current_paths
            mf.set_trained_hash(
                st.session_state.manifest,
                mf.active_hash(st.session_state.manifest),
            )
        st.success("✅ Pipeline actualizado. Los resultados se han refrescado.")
        st.rerun()
