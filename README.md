# BusinessMap — Sistema de IA para Planificación Operativa

Capa de inteligencia complementaria a [BusinessMap](https://businessmap.io) que transforma datos históricos de trabajo en alertas operativas automáticas: predicción de carga por persona, detección de desviaciones por tarea y localización proactiva de cuellos de botella.

> **No sustituye al PM.** No toma decisiones automáticas sobre personas. No modifica BusinessMap ni su flujo. Es un sistema de apoyo a la planificación.

---

## Objetivos

### 1. Predicción de carga futura por responsable
Calcula tasas de entrada/salida por persona en una ventana configurable y proyecta cuántas tareas tendrá cada responsable en los próximos N días, indicando si su carga estará en rango saludable, en riesgo o en sobrecarga.

**Qué no hace BusinessMap:** su Monte Carlo opera sobre throughput agregado del equipo. No proyecta carga individual ni alerta sobre sobrecarga futura por persona.

### 2. Detección de desviación por tarea
Evalúa cada tarea abierta contra benchmarks históricos de duración (por owner, tipo e historial). No predice ETA — detecta cuándo una tarea se desvía significativamente de lo esperado.

**Por qué detección y no ETA:** la dispersión de duraciones en los datos es muy alta (desde minutos hasta meses para el mismo Type Name). Un modelo de ETA con esta granularidad produciría medianas disfrazadas, no predicciones útiles. La detección de desviación es más robusta, más útil operativamente y más honesta metodológicamente.

**Qué no hace BusinessMap:** tiene Aging WIP Chart como visualización pasiva, pero no calcula benchmarks por owner/tipo, no puntúa severidad y no genera alertas automáticas.

### 3. Detección de cuellos de botella
Combina múltiples señales (envejecimiento, estancamiento, acumulación por columna, presión de carga del owner) en una puntuación de riesgo por tarea, por responsable y por fase Kanban.

**Limitación conocida:** el bloqueo explícito es escaso en los datos (10 tarjetas activas). El sistema se apoya en señales implícitas que son más abundantes y fiables.

**Qué no hace BusinessMap:** ofrece CFD y Blocker Charts para interpretación manual, pero no detecta automáticamente, no puntúa severidad y no genera alertas proactivas.

### Relación entre objetivos
Los tres se refuerzan: la desviación de una tarea (obj. 2) alimenta la detección de cuellos de botella (obj. 3). El estado de carga de un responsable (obj. 1) se inyecta como señal en las alertas de sus tareas.

---

## Datos

| Métrica | Valor |
|---------|-------|
| Tarjetas únicas (combinadas) | 1.740 |
| Responsables | 20 |
| Tipos de proyecto | 3 (Externo, Interno, Producto) |
| Tareas cerradas con duración válida | 1.645 |
| Rango temporal | Abril 2024 — Marzo 2026 |
| Fases Kanban con timestamps | 7 |

---

## Estructura del proyecto

```
BusinessMapAtecna/
├── app.py                    # Dashboard Streamlit
├── requirements.txt
├── data/
│   ├── manifest.json         # Fuente de verdad de datasets activos
│   └── raw/                  # Exports de BusinessMap (.xlsx)
├── notebooks/
│   ├── 01_crispdm_baseline   # Baseline histórico (referencia)
│   ├── 02_pipeline_artifacts # Ejecución del pipeline modular
│   ├── 03_forecast_dashboard # Análisis de carga futura
│   ├── 04_bottlenecks        # Alertas y cuellos de botella
│   └── 05_type_segmentation  # Análisis por tipo de proyecto
└── src/
    ├── data_prep.py          # Carga, limpieza, feature engineering
    ├── forecast.py           # Tasas, WIP futuro, escenarios
    ├── bottlenecks.py        # Benchmarks, alertas, scores
    ├── type_segmentation.py  # Análisis segmentado por Type Name
    ├── manifest.py           # Gestión de datasets
    └── pipeline.py           # Orquestación del pipeline
```

La lógica reusable vive en `src/`. Los notebooks se quedan con narrativa CRISP-DM, EDA, visualizaciones e interpretación.

---

## Uso rápido

```python
from src.pipeline import run_baseline_pipeline

artifacts = run_baseline_pipeline(
    workbook_path="data/raw/DatosBusquedaAvanzada20250208.xlsx",
    rate_window_days=60,
    dashboard_horizon_days=5,
    scenario_horizons=(5, 10, 20),
)

# Artefactos principales
forecast = artifacts["forecast_dashboard_export"]   # Carga por owner
alerts = artifacts["task_alerts"]                    # Desviaciones por tarea
owners = artifacts["owner_bottlenecks"]              # Cuellos por responsable
columns = artifacts["column_bottlenecks"]            # Cuellos por fase
```

Dashboard:
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Stack

Python 3.11+ · Pandas · NumPy · Plotly · Streamlit · openpyxl

---

## Estado actual

Prototipo funcional con pipeline completo, dashboard interactivo y gestión de datasets. En fase de reenfoque: mejora del modelado (actualmente heurístico/determinista) y del dashboard (storytelling con datos).
