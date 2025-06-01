import streamlit as st
import pandas as pd
import urllib.parse
import plotly.graph_objs as go
import numpy as np

# Configuración de página
st.set_page_config(
    page_title="Modelo USD Blue | Análisis Económico",
    layout="wide",
    page_icon="📈"
)
st.title("📈 Visualización del Modelo Econométrico del USD Blue")

# Descripción introductoria
st.markdown("""
Este dashboard interactivo permite visualizar el comportamiento histórico del dólar blue en Argentina,  
así como las proyecciones de corto y largo plazo generadas mediante un modelo econométrico.  
Este es un modelo one month ahead (predicción a un mes), por lo que las proyecciones para períodos t+2  
en adelante se realizan bajo el supuesto *ceteris paribus* en las variables macroeconómicas, es decir,  
considerando que estas se mantienen constantes.

---
""")

# ID Google Sheets
SHEET_ID = "1jmzjQvTRWu9Loq_Gpn2SFCvVgo_qPo1X"
HOJAS = ["Datos Originales", "Prediccion_CP", "Prediccion_LP", "Real vs Predicho"]

# Función para cargar hojas
# @st.cache_data(show_spinner=True)
def cargar_hoja(sheet_id, sheet_name):
    sheet_name_encoded = urllib.parse.quote(sheet_name)
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name_encoded}"
    df = pd.read_csv(url)
    # Formateo fechas
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
    
    # Selección hoja
    hoja_sel = st.selectbox("Seleccioná el tipo de gráfico", HOJAS)
    
    # Modo oscuro/claro
    modo_oscuro = st.checkbox("Modo oscuro", value=False)
    
    st.markdown("---")
    st.markdown("📊 [Fuente de datos](https://docs.google.com/spreadsheets/d/1jmzjQvTRWu9Loq_Gpn2SFCvVgo_qPo1X)")

    # Opción para mostrar errores (residuos) solo para "Real vs Predicho"
    mostrar_residuos = False
    if hoja_sel == "Real vs Predicho":
        mostrar_residuos = st.checkbox("Mostrar errores de predicción (residuos)", value=False)

# Carga datos
data = {hoja: cargar_hoja(SHEET_ID, hoja) for hoja in HOJAS}
df = data[hoja_sel]

# Paleta accesible WCAG AA/AAA (azul-verde)
COLOR_PALETA = {
    "real": "#004165",
    "predicho_cp": "#2a9d8f",
    "predicho_lp": "#a3d2ca",
    "intervalo_confianza": "rgba(163, 210, 202, 0.3)",
    "error": "#e76f51",
    "fondo_claro": "#f9f9f9",
    "fondo_oscuro": "#1e1e1e",
    "texto_claro": "#000000",
    "texto_oscuro": "#f7f7f7"  # ← corregido
}

def layout_template(title, modo_oscuro):
    line_color = "#f0f0f0" if modo_oscuro else "#888"

    return dict(
        title=title,
        xaxis=dict(
            title="Fecha",
            showgrid=True,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor=line_color,
            ticks="outside",
            tickformat="%Y-%m",
            tickangle=45,
            dtick="M3"
        ),
        yaxis=dict(
            title="Precio (ARS)",
            showgrid=True,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor=line_color
        ),
        plot_bgcolor=COLOR_PALETA["fondo_oscuro"] if modo_oscuro else COLOR_PALETA["fondo_claro"],
        paper_bgcolor=COLOR_PALETA["fondo_oscuro"] if modo_oscuro else COLOR_PALETA["fondo_claro"],
        font=dict(color=COLOR_PALETA["texto_oscuro"] if modo_oscuro else COLOR_PALETA["texto_claro"]),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)'
        ),
        hovermode='x unified'
    )
# Generar gráfico según hoja seleccionada con Plotly
fig = go.Figure()

if hoja_sel == "Datos Originales":
    df_hist = df[df['MES'] >= '2020-01-01']
    fig.add_trace(go.Scatter(
        x=df_hist['MES'],
        y=df_hist['USD_VENTA'],
        mode='lines+markers',
        name='USD Blue',
        line=dict(color=COLOR_PALETA["real"], width=3),
        marker=dict(size=6),
        hovertemplate='%{x|%Y-%m}: %{y:.2f} ARS<extra></extra>'
    ))
    fig.update_layout(**layout_template("USD Blue histórico (desde 2020)", modo_oscuro))

elif hoja_sel == "Prediccion_CP":
    df_hist = data["Datos Originales"]
    fecha_6m_antes = df_hist['MES'].max() - pd.DateOffset(months=6)
    df_hist_cp = df_hist[df_hist['MES'] >= fecha_6m_antes]
    # Real
    fig.add_trace(go.Scatter(
        x=df_hist_cp['MES'],
        y=df_hist_cp['USD_VENTA'],
        mode='lines+markers',
        name='USD Real',
        line=dict(color=COLOR_PALETA["real"], width=3),
        marker=dict(size=6),
        hovertemplate='%{x|%Y-%m-%d}: %{y:.2f} ARS<extra></extra>'
    ))
    # Predicción CP
    fig.add_trace(go.Scatter(
        x=df['Mes'],
        y=df['USD_Predicho_CP'],
        mode='lines+markers',
        name='Predicción CP',
        line=dict(color=COLOR_PALETA["predicho_cp"], width=3, dash='dash'),
        marker=dict(size=6),
        hovertemplate='%{x|%Y-%m-%d}: %{y:.2f} ARS<extra></extra>'
    ))
    # Intervalo confianza
    fig.add_trace(go.Scatter(
        x=pd.concat([df['Mes'], df['Mes'][::-30]]),
        y=pd.concat([df['IC_Bajo_CP'], df['IC_Alto_CP'][::-30]]),
        fill='toself',
        fillcolor=COLOR_PALETA["intervalo_confianza"],
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=True,
        name='IC 95%'
    ))

    fig.update_layout(**layout_template("Predicción a Corto Plazo", modo_oscuro))

elif hoja_sel == "Prediccion_LP":
    df_hist = data["Datos Originales"]
    df_hist_lp = df_hist[df_hist['MES'] >= '2020-01-01']
    # Real
    fig.add_trace(go.Scatter(
        x=df_hist_lp['MES'],
        y=df_hist_lp['USD_VENTA'],
        mode='lines+markers',
        name='USD Real',
        line=dict(color=COLOR_PALETA["real"], width=3),
        marker=dict(size=6),
        hovertemplate='%{x|%Y-%m}: %{y:.2f} ARS<extra></extra>'
    ))
    # Predicción LP
    fig.add_trace(go.Scatter(
        x=df['Mes'],
        y=df['USD_Predicho_LP'],
        mode='lines+markers',
        name='Predicción LP',
        line=dict(color=COLOR_PALETA["predicho_lp"], width=3, dash='dash'),
        marker=dict(size=6),
        hovertemplate='%{x|%Y-%m}: %{y:.2f} ARS<extra></extra>'
    ))
    # Intervalo confianza
    fig.add_trace(go.Scatter(
        x=pd.concat([df['Mes'], df['Mes'][::-1]]),
        y=pd.concat([df['IC_Bajo_LP'], df['IC_Alto_LP'][::-1]]),
        fill='toself',
        fillcolor=COLOR_PALETA["intervalo_confianza"],
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=True,
        name='IC 95%'
    ))

    fig.update_layout(**layout_template("Predicción a Largo Plazo", modo_oscuro))

elif hoja_sel == "Real vs Predicho":
    # Real y Predicho
    fig.add_trace(go.Scatter(
        x=df['Fecha'],
        y=df['USD_Real'],
        mode='lines+markers',
        name='USD Real',
        line=dict(color=COLOR_PALETA["real"], width=3),
        marker=dict(size=6),
        hovertemplate='%{x|%Y-%m}: %{y:.2f} ARS<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=df['Fecha'],
        y=df['USD_Predicho'],
        mode='lines+markers',
        name='USD Predicho',
        line=dict(color=COLOR_PALETA["predicho_cp"], width=3, dash='dash'),
        marker=dict(size=6),
        hovertemplate='%{x|%Y-%m}: %{y:.2f} ARS<extra></extra>'
    ))

    # Errores (residuos)
    if mostrar_residuos:
        residuos = df['USD_Real'] - df['USD_Predicho']
        fig.add_trace(go.Bar(
            x=df['Fecha'],
            y=residuos,
            name='Error de predicción (residuo)',
            marker_color=COLOR_PALETA["error"],
            opacity=0.6,
            yaxis='y2',
            hovertemplate='%{x|%Y-%m}: %{y:.2f} ARS<extra></extra>'
        ))
        # Añadir segundo eje Y para errores
        fig.update_layout(
            **layout_template("USD Real vs Predicho", modo_oscuro),
            yaxis2=dict(
                title='Error (Residuo)',
                overlaying='y',
                side='right',
                showgrid=False,
                showline=True,
                linecolor='gray'
            )
        )
    else:
        fig.update_layout(**layout_template("USD Real vs Predicho", modo_oscuro))

# Mostrar gráfico
st.plotly_chart(fig, use_container_width=True)

# Botones de descarga CSV y PNG
import io
csv_buffer = df.to_csv(index=False).encode("utf-8")
img_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)

with st.sidebar:
    st.download_button("⬇️ Descargar CSV", data=csv_buffer, file_name=f"{hoja_sel.replace(' ', '_')}.csv", mime="text/csv")
    st.download_button("⬇️ Descargar imagen PNG", data=img_bytes, file_name=f"{hoja_sel.replace(' ', '_')}.png", mime="image/png")

# Sección colapsable "Sobre el modelo"
with st.expander("📖 Sobre el modelo"):
    st.markdown("""
    **Metodología del modelo econométrico:**

    - **Tipo de regresión:** Regresión lineal múltiple con variables macroeconómicas.
    - **Variables incluidas:** IPC, Reservas Internacionales, BADLAR, Riesgo País, MEP.
    - **Supuestos clave:**
      - Linealidad entre variables y precio USD blue.
      - Variables macroeconómicas consideradas exógenas.
      - Independencia y homocedasticidad de residuos.
      - Modelo one-month-ahead (predicción a un mes).

    El modelo se ajusta con datos históricos mensuales, y se valida con métricas de error  
    como MAE y RMSE. Las predicciones de largo plazo asumen estabilidad en las variables macro.
    """)

# Footer
st.markdown("---")
st.markdown("© 2025 Santiago Wickham | Proyecto económico - Datos: Fuentes oficiales y Google Sheets")
st.markdown(""" 

🔗 [LinkedIn](https://www.linkedin.com/in/santiagowickham/)  
🐙 [GitHub](https://github.com/SantiagoWickham)

""")
