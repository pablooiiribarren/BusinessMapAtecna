# Notebooks

Estructura recomendada a partir de ahora:

- `01_crispdm_baseline.ipynb`
  Baseline historico completo. Mantener como referencia del trabajo exploratorio original.
- `02_pipeline_artifacts.ipynb`
  Ejecuta el pipeline modular y deja todos los artefactos base listos.
- `03_forecast_dashboard.ipynb`
  Analisis enfocado en carga futura, escenarios y tablas para dashboard.
- `04_bottlenecks.ipynb`
  Analisis enfocado en alertas, owners, estados y tipos con riesgo operativo.
- `05_type_segmentation.ipynb`
  Reconstruye forecast y bottlenecks por `Type Name` para aislar `PROYECTO EXTERNO`, `PROYECTO INTERNO` y `PRODUCTO`.

Recomendacion de uso:

1. Mantener `01_crispdm_baseline.ipynb` como cuaderno historico y de auditoria.
2. Continuar el trabajo operativo en `02`, `03` y `04`.
3. Usar `05` para comparacion por tipo sin mezclar comportamientos entre `EXTERNO`, `INTERNO` y `PRODUCTO`.
4. Construir nuevas iteraciones sobre los notebooks modulares, no sobre el baseline original.
