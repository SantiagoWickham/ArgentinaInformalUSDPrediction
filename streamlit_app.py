import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import urllib.parse
import numpy as np

# Configuración de página
st.set_page_config(page_title="Modelo USD Blue | Análisis Económico", layout="wide")
st.title("📈 Visualización del Modelo Econométrico del USD Blue")

# Descripción introductoria
st.markdown("""
Este dashboard interactivo permite visualizar el comportamiento histórico del dólar blue en Argentina, 
así como las proyecciones de corto y largo plazo generadas mediante un modelo econométrico.  
Las bandas representan intervalos de confianza del 95%.

---  
""")

# Carga de datos
sheet_id = "1jmzjQvTRWu9Loq_Gpn2SFCvVgo_qPo1X"

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
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/Flag_of_Argentina.svg/800px-Flag_of_Argentina.svg.png", width=100)
    st.header("⚙️ Configuración")
    hojas = ["Datos Originales", "Prediccion_CP", "Prediccion_LP", "Real vs Predicho"]
    hoja_sel = st.selectbox("Seleccioná el tipo de gráfico", hojas)
    st.markdown("---")
    st.markdown("📊 [Fuente de datos](https://docs.google.com/spreadsheets/d/1jmzjQvTRWu9Loq_Gpn2SFCvVgo_qPo1X)")

# Cargar hojas
data = {hoja: cargar_hoja(sheet_id, hoja) for hoja in hojas}
df = data[hoja_sel]

# Estética de visualización
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(14, 6))

# Gráfico según selección
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
    primer_mes = df['Mes'].iloc[0]
    valor_actual = df['USD_Predicho_CP'].iloc[0]
    segundo_mes = df['Mes'].iloc[1]
    ic_bajo_segundo = df['IC_Bajo_CP'].iloc[1]
    ic_alto_segundo = df['IC_Alto_CP'].iloc[1]
    primer_mes_num = mdates.date2num(primer_mes)
    segundo_mes_num = mdates.date2num(segundo_mes)
    ax.plot([primer_mes_num, segundo_mes_num], [valor_actual, ic_bajo_segundo], color='#bde7b7', linewidth=0.5)
    ax.plot([primer_mes_num, segundo_mes_num], [valor_actual, ic_alto_segundo], color='#bde7b7', linewidth=0.5)
    ax.fill_between([primer_mes_num, segundo_mes_num], [valor_actual, ic_bajo_segundo], [valor_actual, ic_alto_segundo], color='#bde7b7', alpha=0.25)
    fechas_ic_num = mdates.date2num(df['Mes'].iloc[1:])
    ax.fill_between(fechas_ic_num, df['IC_Bajo_CP'].iloc[1:], df['IC_Alto_CP'].iloc[1:], color='#bde7b7', alpha=0.25, label='IC 95%')
    ax.set_title("Predicción a Corto Plazo", fontsize=16)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio (ARS)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

elif hoja_sel == "Prediccion_LP":
    df_hist = data["Datos Originales"]
    df_hist_lp = df_hist[df_hist['MES'] >= '2020-01-01']
    ax.plot(df_hist_lp['MES'], df_hist_lp['USD_VENTA'], label='USD Real', color='#003f5c', linewidth=2)
    ax.plot(df['Mes'], df['USD_Predicho_LP'], label='Predicción LP', color='#7bcf6f', linewidth=2, linestyle='--')
    primer_mes = df['Mes'].iloc[0]
    valor_actual = df['USD_Predicho_LP'].iloc[0]
    segundo_mes = df['Mes'].iloc[1]
    ic_bajo_segundo = df['IC_Bajo_LP'].iloc[1]
    ic_alto_segundo = df['IC_Alto_LP'].iloc[1]
    primer_mes_num = mdates.date2num(primer_mes)
    segundo_mes_num = mdates.date2num(segundo_mes)
    ax.plot([primer_mes_num, segundo_mes_num], [valor_actual, ic_bajo_segundo], color='#bde7b7', linewidth=0.5)
    ax.plot([primer_mes_num, segundo_mes_num], [valor_actual, ic_alto_segundo], color='#bde7b7', linewidth=0.5)
    ax.fill_between([primer_mes_num, segundo_mes_num], [valor_actual, ic_bajo_segundo], [valor_actual, ic_alto_segundo], color='#bde7b7', alpha=0.25)
    fechas_ic_num = mdates.date2num(df['Mes'].iloc[1:])
    ax.fill_between(fechas_ic_num, df['IC_Bajo_LP'].iloc[1:], df['IC_Alto_LP'].iloc[1:], color='#bde7b7', alpha=0.25, label='IC 95%')
    ax.set_title("Predicción a Largo Plazo", fontsize=16)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio (ARS)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

elif hoja_sel == "Real vs Predicho":
    ax.plot(df['Fecha'], df['USD_Real'], label='USD Real', color='black', linewidth=2)
    ax.plot(df['Fecha'], df['USD_Predicho'], label='USD Predicho', color='red', linewidth=2, linestyle='--')
    ax.set_title("Comparación: Real vs Predicho", fontsize=16)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio (ARS)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

# Estilo general
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
sns.despine()
st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("📍 Desarrollado por Santiago Wickham | Lic. en Economía y Finanzas | Datos desde Google Sheets | Visualización con Streamlit + Matplotlib")
