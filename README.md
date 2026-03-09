# BusinessMapAtecna

## 🎯 Objetivo 1 — Predicción de carga futura

Forecast del WIP por responsable en horizonte H (ej: 5, 10, 20 días).

Esto requiere:

-   Modelo de arrivals rate (nuevas tareas asignadas)

-   Modelo de completions rate (velocidad histórica de cierre)

## 🎯 Objetivo 2 — ETA real por tarea

Estimación individual del tiempo restante hasta cierre.

Esto requiere:

-   Modelo de regresión sobre duración

-   Features de contexto: owner, tipo, edad, estado, histórico similar

## 🎯 Objetivo 3 — Detección de cuellos de botella

Detección temprana de acumulación anómala.

Esto requiere:

-   Métricas de envejecimiento

-   Distribución por estados

-   Análisis de dependencias (Links)

-   Señales de riesgo agregadas

## ⚠️ Punto clave:

El objetivo 1 y 3 se apoyan en el 2.

Si sabes cuánto tardan realmente las tareas, puedes:

-   estimar carga futura mejor

-   detectar desviaciones antes

## Estructura recomendada

El notebook no deberia ser la unica fuente de logica. La estructura recomendada a partir de ahora es:

- `src/data_prep.py`: carga, limpieza, parseo de fechas y features base
- `src/forecast.py`: calculo de tasas, forecast de WIP y escenarios
- `src/bottlenecks.py`: benchmarks, alertas y agregados de cuellos de botella
- `src/type_segmentation.py`: reconstruccion y comparacion por `Type Name`
- `src/pipeline.py`: orquestacion del pipeline base

Uso minimo desde notebook:

```python
from pathlib import Path
import sys

project_root = Path.cwd().resolve()
if project_root.name == "notebooks":
    project_root = project_root.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipeline import run_baseline_pipeline

artifacts = run_baseline_pipeline(
    workbook_path=PATH_XLSX,
    rate_window_days=60,
    dashboard_horizon_days=5,
    scenario_horizons=(5, 10, 20),
)

bm = artifacts["bm"]
forecast_dashboard_export = artifacts["forecast_dashboard_export"]
forecast_scenarios = artifacts["forecast_scenarios"]
task_alerts = artifacts["task_alerts"]
owner_bottlenecks = artifacts["owner_bottlenecks"]
column_bottlenecks = artifacts["column_bottlenecks"]
type_bottlenecks = artifacts["type_bottlenecks"]
```

El notebook deberia quedarse con:

- narrativa CRISP-DM
- checks de calidad y EDA
- visualizaciones
- interpretacion de resultados

La logica reusable y estable deberia vivir en `src/`.
