import streamlit as st
import pandas as pd
import numpy as np
import urllib.parse
import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go

# CONFIGURACIÓN INICIAL
st.set_page_config(page_title="Modelo USD Blue | Análisis Económico", layout="wide", page_icon="📈")

# TÍTULO Y DESCRIPCIÓN
st.title("📈 Visualización del Modelo Econométrico del USD Blue")
st.markdown("""
Este dashboard interactivo permite visualizar el comportamiento histórico del dólar blue en Argentina, 
así como las proyecciones de corto y largo plazo generadas mediante un modelo econométrico.  
Este es un modelo one month ahead (predicción a un mes), por lo que las proyecciones para períodos t+2
en adelante se realizan bajo el supuesto *ceteris paribus*, considerando las variables macroeconómicas constantes.
---
""")

# ID de la Google Sheet
sheet_id = "1jmzjQvTRWu9Loq_Gpn2SFCvVgo_qPo1X"

# FUNCIÓN DE CARGA DE DATOS
@st.cache_data(ttl=600)
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

# SIDEBAR
with st.sidebar:
    st.image("https://raw.githubusercontent.com/SantiagoWickham/ArgentinaInformalUSDPrediction/main/logo.jpg", width=100)
    st.header("⚙️ Configuración")
    hoja_sel = st.selectbox("Seleccioná el tipo de gráfico", ["Datos Originales", "Prediccion_CP", "Prediccion_LP", "Real vs Predicho"])
    actualizar = st.slider("⏱️ Refrescar cada X segundos", min_value=0, max_value=300, step=10, value=0)
    st.markdown("---")
    st.markdown("📊 [Fuente de datos](https://docs.google.com/spreadsheets/d/1jmzjQvTRWu9Loq_Gpn2SFCvVgo_qPo1X)")

# AUTO-REFRESH
if actualizar > 0:
    st.experimental_rerun()

# CARGA DE DATOS
data = {hoja: cargar_hoja(sheet_id, hoja) for hoja in ["Datos Originales", "Prediccion_CP", "Prediccion_LP", "Real vs Predicho"]}
df = data[hoja_sel]

# MODO GRÁFICO PLOTLY (INTERACTIVO CON TOOLTIPS)
if hoja_sel == "Datos Originales":
    df_hist = df[df['MES'] >= '2020-01-01']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_hist['MES'], y=df_hist['USD_VENTA'],
                             mode='lines+markers',
                             name='USD Blue',
                             line=dict(color='#003f5c')))
    fig.update_layout(title="USD Blue histórico (desde 2020)",
                      xaxis_title="Fecha",
                      yaxis_title="Precio (ARS)",
                      template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white")

elif hoja_sel == "Prediccion_CP":
    df_hist = data["Datos Originales"]
    fecha_6m_antes = df_hist['MES'].max() - pd.DateOffset(months=6)
    df_hist_cp = df_hist[df_hist['MES'] >= fecha_6m_antes]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_hist_cp['MES'], y=df_hist_cp['USD_VENTA'], mode='lines', name="USD Real", line=dict(color='#2f4b7c')))
    fig.add_trace(go.Scatter(x=df['Mes'], y=df['USD_Predicho_CP'], mode='lines+markers', name="Predicción CP", line=dict(color='#2f7c5e', dash="dash")))
    fig.add_trace(go.Scatter(x=df['Mes'], y=df['IC_Bajo_CP'], name='IC Inferior', line=dict(color='rgba(0,0,0,0)'), showlegend=False))
    fig.add_trace(go.Scatter(x=df['Mes'], y=df['IC_Alto_CP'], name='IC Superior', fill='tonexty', fillcolor='rgba(189,231,183,0.3)', line=dict(color='rgba(0,0,0,0)'), showlegend=True))

    fig.update_layout(title="Predicción a Corto Plazo", xaxis_title="Fecha", yaxis_title="Precio (ARS)",
                      template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white")

elif hoja_sel == "Prediccion_LP":
    df_hist = data["Datos Originales"]
    df_hist_lp = df_hist[df_hist['MES'] >= '2020-01-01']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_hist_lp['MES'], y=df_hist_lp['USD_VENTA'], mode='lines', name="USD Real", line=dict(color='#003f5c')))
    fig.add_trace(go.Scatter(x=df['Mes'], y=df['USD_Predicho_LP'], mode='lines+markers', name="Predicción LP", line=dict(color='#7bcf6f', dash='dash')))
    fig.add_trace(go.Scatter(x=df['Mes'], y=df['IC_Bajo_LP'], name='IC Inferior', line=dict(color='rgba(0,0,0,0)'), showlegend=False))
    fig.add_trace(go.Scatter(x=df['Mes'], y=df['IC_Alto_LP'], name='IC Superior', fill='tonexty', fillcolor='rgba(189,231,183,0.3)', line=dict(color='rgba(0,0,0,0)'), showlegend=True))

    fig.update_layout(title="Predicción a Largo Plazo", xaxis_title="Fecha", yaxis_title="Precio (ARS)",
                      template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white")

elif hoja_sel == "Real vs Predicho":
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Fecha'], y=df['USD_Real'], mode='lines+markers', name="USD Real", line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df['Fecha'], y=df['USD_Predicho'], mode='lines+markers', name="USD Predicho", line=dict(color='red', dash='dash')))

    fig.update_layout(title="Comparación: Real vs Predicho",
                      xaxis_title="Fecha", yaxis_title="Precio (ARS)",
                      template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white")

# RENDER GRÁFICO
st.plotly_chart(fig, use_container_width=True)

# DESCARGAS CSV + PNG
csv_buffer = df.to_csv(index=False).encode("utf-8")
img_buffer = io.BytesIO()
fig.write_image(img_buffer, format="png")
img_buffer.seek(0)

with st.sidebar:
    st.download_button("⬇️ Descargar CSV", data=csv_buffer, file_name=f"{hoja_sel}.csv", mime="text/csv")
    st.download_button("🖼️ Descargar gráfico PNG", data=img_buffer, file_name=f"grafico_{hoja_sel}.png", mime="image/png")

# FOOTER
st.markdown("---")
st.markdown("""
📍 **Desarrollado por:** Santiago Wickham  
Estudiante de Lic. en Economía y Finanzas  

🔗 [LinkedIn](https://www.linkedin.com/in/santiagowickham/)  
🐙 [GitHub](https://github.com/SantiagoWickham)
""")
