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