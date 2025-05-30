 Predicción del Dólar Informal en Argentina usando Estabilidad Macroeconómica Sintética

Este proyecto integra ciencia de datos y análisis económico para construir un modelo de predicción del dólar informal argentino (dólar blue). Utiliza un índice sintético de estabilidad macroeconómica generado con Análisis de Componentes Principales (PCA) sobre variables económicas públicas, levantadas desde una hoja de cálculo en Google Drive. La visualización final se presenta en Power BI.

---

 Objetivo

El objetivo es demostrar cómo se puede combinar Python, datos económicos abiertos y herramientas de visualización para modelar fenómenos complejos como la evolución del dólar informal en Argentina, explorando la relación entre la estabilidad macroeconómica y la dinámica del tipo de cambio paralelo.

---

 Herramientas utilizadas

- Python + Jupyter Notebook  
  · pandas, numpy, matplotlib, seaborn  
  · sklearn (para PCA y regresión lineal)

- Google Sheets (Drive)
  · Fuente online y dinámica de los datos macroeconómicos

- Power BI
  · Visualización interactiva de los resultados predictivos

---

 Fuente de datos

Los datos macroeconómicos se levantan dinámicamente desde una hoja de cálculo pública en Google Sheets:

 URL (https://docs.google.com/spreadsheets/d/1gWaV7dbwImFO6NrfTO08QmZGZIdHBsOdyweaR3BJbYY/edit?usp=drive_link):  

Desde el script de Python, los datos se leen directamente con:

sheet_id = "1gWaV7dbwImFO6NrfTO08QmZGZIdHBsOdyweaR3BJbYY"
sheet_name = "Datos%20Prediccion%20USD%20Blue"  # <- espacios reemplazados
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
df = pd.read_csv(url)