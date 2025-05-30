import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Configuración de página
st.set_page_config(
    page_title="Predicción del Dólar Blue en Argentina",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos de accesibilidad (WCAG AA/AAA)
st.markdown("""
    <style>
    body {
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        background-color: #F9F9F9;
        color: #222;
    }
    h1, h2, h3, h4 {
        color: #2C3E50;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
        border-bottom: 1px dotted black;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        padding: 5px;
        border-radius: 6px;
        position: absolute;
        z-index: 1;
        bottom: 125%; 
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.title("📈 Predicción del Dólar Blue en Argentina")

# Selector de tema visual
modo = st.sidebar.radio("🎨 Elegir tema", ["Claro", "Oscuro"])
if modo == "Oscuro":
    st.markdown("""
        <style>
        .main {
            background-color: #1E1E1E;
            color: #F5F5F5;
        }
        h1, h2, h3, h4 {
            color: #ECF0F1;
        }
        </style>
    """, unsafe_allow_html=True)

# Cargar datos (ejemplo)
@st.cache_data
def cargar_datos():
    # Aquí deberías cargar tus datos reales
    fechas = pd.date_range("2023-01-01", periods=24, freq='M')
    df = pd.DataFrame({
        "Fecha": fechas,
        "Dólar Blue Observado": 300 + (fechas.month * 10) + (fechas.month % 3) * 20,
        "Dólar Blue Proyectado": 310 + (fechas.month * 11),
        "Error Absoluto": abs(((fechas.month * 11 + 310) - (fechas.month * 10 + 300)))
    })
    return df

df = cargar_datos()

# Visualización interactiva
fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(
    x=df["Fecha"], y=df["Dólar Blue Observado"],
    mode="lines+markers",
    name="Observado",
    line=dict(color="#1f77b4")
))
fig_pred.add_trace(go.Scatter(
    x=df["Fecha"], y=df["Dólar Blue Proyectado"],
    mode="lines+markers+text",
    name="Proyectado",
    line=dict(color="#ff7f0e"),
    text=[f"${val:.0f}" for val in df["Dólar Blue Proyectado"]],
    textposition="top center",
    hovertemplate="Fecha: %{x|%b %Y}<br>Valor proyectado: $%{y:.2f}"
))
fig_pred.update_layout(
    title="Evolución del Dólar Blue: Observado vs Proyectado",
    xaxis_title="Fecha",
    yaxis_title="ARS",
    template="plotly_dark" if modo == "Oscuro" else "plotly_white"
)
st.plotly_chart(fig_pred, use_container_width=True)

# Análisis de errores
fig_error = px.bar(
    df,
    x="Fecha",
    y="Error Absoluto",
    title="Error absoluto mensual de la proyección",
    labels={"Error Absoluto": "Error ($)", "Fecha": "Mes"},
    template="plotly_dark" if modo == "Oscuro" else "plotly_white",
    color_discrete_sequence=["#e74c3c"]
)
st.plotly_chart(fig_error, use_container_width=True)

# Expandible con metodología del modelo
with st.expander("🧠 Metodología del Modelo"):
    st.markdown("""
    Este modelo econométrico busca predecir el valor del **dólar blue** en Argentina utilizando variables macroeconómicas relevantes como:

    - **IPC (Inflación)** mensual
    - **Reservas Internacionales del BCRA**
    - **Tasa BADLAR**
    - **Base Monetaria (M2)**
    - **Tipo de cambio MEP**
    - **Resultado Primario (RP)**

    Se construyó un índice sintético de estabilidad macroeconómica usando análisis de componentes principales (**PCA**) para captar dinámicas latentes.

    Luego se aplicó **regresión lineal múltiple**, validación temporal y análisis de residuos. Se reportan errores como **MAE** y **RMSE**, además de predicciones mensuales con **intervalos de confianza**.
    """)

# Footer
st.markdown("---")
st.markdown("© 2025 **Santiago Wickham** | Proyecto económico - Datos: Fuentes oficiales")
st.markdown(""" 

🔗 [LinkedIn](https://www.linkedin.com/in/santiagowickham/)  
🐙 [GitHub](https://github.com/SantiagoWickham)

""") 
