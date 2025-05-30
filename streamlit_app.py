import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import urllib.parse
import numpy as np
import io

# Configuración de la página
st.set_page_config(
    page_title="Modelo USD Blue | Análisis Económico",
    layout="wide",
    page_icon="📈"
)

# Título principal
st.title("📈 Visualización del Modelo Econométrico del USD Blue")

# Introducción
st.markdown("""
Este dashboard interactivo permite visualizar el comportamiento histórico del dólar blue en Argentina,
así como las proyecciones de corto y largo plazo generadas mediante un modelo econométrico.

El modelo realiza predicciones mensuales (one month ahead), por lo que las proyecciones para períodos t+2 en adelante
se basan en el supuesto ceteris paribus, manteniendo constantes las variables macroeconómicas involucradas.
""")

st.markdown("---")

# ID de Google Sheets
SHEET_ID = "1jmzjQvTRWu9Loq_Gpn2SFCvVgo_qPo1X"

# Función para cargar datos desde una hoja específica
@st.cache_data(show_spinner=True)
def cargar_hoja(sheet_id: str, sheet_name: str) -> pd.DataFrame:
    encoded_name = urllib.parse.quote(sheet_name)
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={encoded_name}"
    df = pd.read_csv(url)

    date_col = {
        "Datos Originales": "MES",
        "Prediccion_CP": "Mes",
        "Prediccion_LP": "Mes",
        "Real vs Predicho": "Fecha"
    }.get(sheet_name, None)

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values(date_col)

    return df

# Sidebar: Configuración
with st.sidebar:
    st.image("https://raw.githubusercontent.com/SantiagoWickham/ArgentinaInformalUSDPrediction/main/logo.jpg", width=100)
    st.header("⚙️ Configuración")
    opciones = ["Datos Originales", "Prediccion_CP", "Prediccion_LP", "Real vs Predicho"]
    hoja_sel = st.selectbox("Seleccioná el tipo de gráfico", opciones)
    st.markdown("---")
    st.markdown("📊 [Fuente de datos](https://docs.google.com/spreadsheets/d/1jmzjQvTRWu9Loq_Gpn2SFCvVgo_qPo1X)")

# Carga de todos los datasets
data = {hoja: cargar_hoja(SHEET_ID, hoja) for hoja in opciones}
df = data[hoja_sel]

# Configuración general del gráfico
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(14, 6))

# Gráficos por hoja
if hoja_sel == "Datos Originales":
    df_plot = df[df['MES'] >= '2020-01-01']
    ax.plot(df_plot['MES'], df_plot['USD_VENTA'], label='USD Blue', color='#003f5c', linewidth=2)
    ax.set_title("USD Blue histórico (desde 2020)", fontsize=16)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio (ARS)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

elif hoja_sel == "Prediccion_CP":
    df_hist = data["Datos Originales"]
    fecha_inicio = df_hist['MES'].max() - pd.DateOffset(months=6)
    df_hist = df_hist[df_hist['MES'] >= fecha_inicio]
    ax.plot(df_hist['MES'], df_hist['USD_VENTA'], label='USD Real', color='#2f4b7c', linewidth=2)
    ax.plot(df['Mes'], df['USD_Predicho_CP'], label='Predicción CP', color='#2f7c5e', linewidth=2, linestyle='--')
    ax.fill_between(df['Mes'], df['IC_Bajo_CP'], df['IC_Alto_CP'], color='#bde7b7', alpha=0.3, label='IC 95%')
    primer_pred = df[df['Delta_CP'] != 0].iloc[0]
    ax.annotate(
        f'{primer_pred["USD_Predicho_CP"]:.2f} ARS\n{primer_pred["Mes"].strftime("%Y-%m-%d")}',
        xy=(primer_pred['Mes'], primer_pred['USD_Predicho_CP']),
        xytext=(primer_pred['Mes'], primer_pred['USD_Predicho_CP'] + 40),
        arrowprops=dict(facecolor='#a3d9a5', arrowstyle='->'),
        fontsize=12, ha='center', color='#2f7c5e'
    )
    ax.set_title("Predicción a Corto Plazo", fontsize=16)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

elif hoja_sel == "Prediccion_LP":
    df_hist = data["Datos Originales"]
    df_hist = df_hist[df_hist['MES'] >= '2020-01-01']
    ax.plot(df_hist['MES'], df_hist['USD_VENTA'], label='USD Real', color='#003f5c', linewidth=2)
    ax.plot(df['Mes'], df['USD_Predicho_LP'], label='Predicción LP', color='#7bcf6f', linewidth=2, linestyle='--')
    ax.fill_between(df['Mes'], df['IC_Bajo_LP'], df['IC_Alto_LP'], color='#bde7b7', alpha=0.3, label='IC 95%')
    primer_pred = df[df['Delta_LP'] != 0].iloc[0]
    ax.annotate(
        f'{primer_pred["USD_Predicho_LP"]:.2f} ARS\n{primer_pred["Mes"].strftime("%Y-%m-%d")}',
        xy=(primer_pred['Mes'], primer_pred['USD_Predicho_LP']),
        xytext=(primer_pred['Mes'], primer_pred['USD_Predicho_LP'] - 300),
        arrowprops=dict(facecolor='#a3d9a5', arrowstyle='->'),
        fontsize=12, ha='center', color='#7bcf6f'
    )
    ax.set_title("Predicción a Largo Plazo", fontsize=16)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

elif hoja_sel == "Real vs Predicho":
    ax.plot(df['Fecha'], df['USD_Real'], label='USD Real', color='black', linewidth=2)
    ax.plot(df['Fecha'], df['USD_Predicho'], label='USD Predicho', color='red', linewidth=2, linestyle='--')
    ax.set_title("Comparación: Real vs Predicho", fontsize=16)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio (ARS)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fila_junio = df[df['Fecha'] == pd.Timestamp('2025-06-30')]
    if not fila_junio.empty:
        valor_junio = fila_junio['USD_Predicho'].values[0]
        ax.annotate(f'Junio: {valor_junio:.2f}',
                    xy=(fila_junio['Fecha'].values[0], valor_junio),
                    xytext=(fila_junio['Fecha'].values[0], valor_junio + 30),
                    arrowprops=dict(facecolor='black', arrowstyle='->'),
                    fontsize=12, ha='center')

# Estética general del gráfico
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
sns.despine()
st.pyplot(fig)

# Botones de descarga
csv_data = df.to_csv(index=False).encode("utf-8")
img_buffer = io.BytesIO()
fig.savefig(img_buffer, format="png", bbox_inches='tight')
img_buffer.seek(0)

with st.sidebar:
    st.download_button("⬇️ Descargar CSV", data=csv_data, file_name=f"{hoja_sel}.csv", mime="text/csv")
    st.download_button("🖼️ Descargar gráfico PNG", data=img_buffer, file_name=f"grafico_{hoja_sel}.png", mime="image/png")

# Pie de página
st.markdown("""
---
📍 **Desarrollado por:** Santiago Wickham  
Estudiante de Lic. en Economía y Finanzas  

🔗 [LinkedIn](https://www.linkedin.com/in/santiagowickham/)  
🐙 [GitHub](https://github.com/SantiagoWickham)
""")
