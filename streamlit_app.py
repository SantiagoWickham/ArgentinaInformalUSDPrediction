import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import urllib.parse
import numpy as np
import io

# Configuración de página
st.set_page_config(page_title="Modelo USD Blue | Análisis Económico", layout="wide", page_icon="📈")
st.title("📈 Visualización del Modelo Econométrico del USD Blue")

# Descripción introductoria
st.markdown("""
Este dashboard interactivo permite visualizar el comportamiento histórico del dólar blue en Argentina, 
así como las proyecciones de corto y largo plazo generadas mediante un modelo econométrico.  
Este es un modelo one month ahead (predicción a un mes), por lo que las proyecciones para períodos t+2
en adelante se realizan bajo el supuesto ceteris paribus en las variables macroeconómicas, es decir, 
considerando que estas se mantienen constantes.

---
""")

# ID de Google Sheets
sheet_id = "1jmzjQvTRWu9Loq_Gpn2SFCvVgo_qPo1X"

# Función para cargar hojas
@st.cache_data(show_spinner=True)
def cargar_hoja(sheet_id, sheet_name):
    sheet_name_encoded = urllib.parse.quote(sheet_name)
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name_encoded}"
    df = pd.read_csv(url)
    
    if sheet_name == "Datos Originales":
        df['MES'] = pd.to_datetime(df['MES'], errors='coerce')
        df = df.sort_values('MES')
    elif sheet_name in ["Prediccion_CP", "Prediccion_LP"]:
        df['Mes'] = pd.to_datetime(df['Mes'], errors='coerce')
        df = df.sort_values('Mes')
    elif sheet_name == "Real vs Predicho":
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df = df.sort_values('Fecha')
    return df

# Sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/SantiagoWickham/ArgentinaInformalUSDPrediction/main/logo.jpg", width=100)
    st.header("⚙️ Configuración")
    hojas = ["Datos Originales", "Prediccion_CP", "Prediccion_LP", "Real vs Predicho"]
    hoja_sel = st.selectbox("Seleccioná el tipo de gráfico", hojas)
    st.markdown("---")
    st.markdown("📊 [Fuente de datos](https://docs.google.com/spreadsheets/d/1jmzjQvTRWu9Loq_Gpn2SFCvVgo_qPo1X)")

# Carga los datos después de definir la hoja seleccionada
data = {hoja: cargar_hoja(sheet_id, hoja) for hoja in hojas}
df = data[hoja_sel]

# Estilo visual
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(14, 6))

# Gráfico según hoja seleccionada
if hoja_sel == "Datos Originales":
    df_hist = df[df['MES'] >= '2020-01-01']
    ax.plot(df_hist['MES'], df_hist['USD_VENTA'], label='USD Blue', color='#003f5c', linewidth=2)
    ax.set_title("USD Blue histórico (desde 2020)", fontsize=16)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio (ARS)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

elif hoja_sel == "Prediccion_CP":
    df_hist = data["Datos Originales"]
    fecha_6m_antes = df_hist['MES'].max() - pd.DateOffset(months=6)
    df_hist_cp = df_hist[df_hist['MES'] >= fecha_6m_antes]
    ax.plot(df_hist_cp['MES'], df_hist_cp['USD_VENTA'], label='USD Real', color='#2f4b7c', linewidth=2)
    ax.plot(df['Mes'], df['USD_Predicho_CP'], label='Predicción CP', color='#2f7c5e', linewidth=2, linestyle='--')
    ax.fill_between(df['Mes'], df['IC_Bajo_CP'], df['IC_Alto_CP'], color='#bde7b7', alpha=0.3, label='IC 95%')
    ax.set_title("Predicción a Corto Plazo", fontsize=16)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio (ARS)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    # Etiqueta con primer valor predicho CP (excluyendo la fila "Actual" si tiene Delta=0)
    primer_pred_cp = df[df['Delta_CP'] != 0].iloc[0]  # Tomar la primera fila con Delta distinto de 0
    ax.annotate(
        f'1er Predicho CP:\n{primer_pred_cp["USD_Predicho_CP"]:.2f} ARS\n{primer_pred_cp["Mes"].strftime("%Y-%m-%d")}',
        xy=(primer_pred_cp['Mes'], primer_pred_cp['USD_Predicho_CP']),
        xytext=(primer_pred_cp['Mes'], primer_pred_cp['USD_Predicho_CP'] + 40),  # Ajusta vertical
        arrowprops=dict(facecolor='#a3d9a5', arrowstyle='->'),
        fontsize=12,
        ha='center',
        color='#a3d9a5'
    )

elif hoja_sel == "Prediccion_LP":
    df_hist = data["Datos Originales"]
    df_hist_lp = df_hist[df_hist['MES'] >= '2020-01-01']
    ax.plot(df_hist_lp['MES'], df_hist_lp['USD_VENTA'], label='USD Real', color='#003f5c', linewidth=2)
    ax.plot(df['Mes'], df['USD_Predicho_LP'], label='Predicción LP', color='#7bcf6f', linewidth=2, linestyle='--')
    ax.fill_between(df['Mes'], df['IC_Bajo_LP'], df['IC_Alto_LP'], color='#bde7b7', alpha=0.3, label='IC 95%')
    ax.set_title("Predicción a Largo Plazo", fontsize=16)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio (ARS)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    # Etiqueta con primer valor predicho LP (excluyendo fila "Actual" con Delta=0)
    primer_pred_lp = df[df['Delta_LP'] != 0].iloc[0]
    ax.annotate(
        f'1er Predicho LP:\n{primer_pred_lp["USD_Predicho_LP"]:.2f} ARS\n{primer_pred_lp["Mes"].strftime("%Y-%m-%d")}',
        xy=(primer_pred_lp['Mes'], primer_pred_lp['USD_Predicho_LP']),
        xytext=(primer_pred_lp['Mes'], primer_pred_lp['USD_Predicho_LP'] - 100),
        arrowprops=dict(facecolor='#a3d9a5', arrowstyle='->'),
        fontsize=12,
        ha='center',
        color='#a3d9a5'
    )

elif hoja_sel == "Real vs Predicho":
    ax.plot(df['Fecha'], df['USD_Real'], label='USD Real', color='black', linewidth=2)
    ax.plot(df['Fecha'], df['USD_Predicho'], label='USD Predicho', color='red', linewidth=2, linestyle='--')
    ax.set_title("Comparación: Real vs Predicho", fontsize=16)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio (ARS)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
     # Agregar etiqueta para valor predicho junio 2025
    fecha_junio = pd.Timestamp('2025-06-30 00:00:00')
    fila_junio = df[df['Mes'] == fecha_junio]
    if not fila_junio.empty:
        valor_junio = fila_junio['USD_Predicho_LP'].values[0]
        ax.annotate(f'Junio: {valor_junio:.2f}',
                    xy=(fecha_junio, valor_junio),
                    xytext=(fecha_junio, valor_junio + 30),  # Ajusta según escala vertical
                    arrowprops=dict(facecolor='black', arrowstyle='->'),
                    fontsize=12,
                    ha='center')

# Estilo general
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
sns.despine()
st.pyplot(fig)

# BOTÓN DE DESCARGA CSV y PNG
csv_buffer = df.to_csv(index=False).encode("utf-8")
img_buffer = io.BytesIO()
fig.savefig(img_buffer, format="png", bbox_inches='tight')
img_buffer.seek(0)

with st.sidebar:
    st.download_button("⬇️ Descargar CSV", data=csv_buffer, file_name=f"{hoja_sel}.csv", mime="text/csv")
    st.download_button("🖼️ Descargar gráfico PNG", data=img_buffer, file_name=f"grafico_{hoja_sel}.png", mime="image/png")

# Footer
st.markdown("---")
st.markdown("""
📍 **Desarrollado por:** Santiago Wickham  
Estudiante de Lic. en Economía y Finanzas  

🔗 [LinkedIn](https://www.linkedin.com/in/santiagowickham/)  
🐙 [GitHub](https://github.com/SantiagoWickham)
""")
