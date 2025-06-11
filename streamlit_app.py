import streamlit as st
import pandas as pd
import urllib.parse
import plotly.graph_objs as go
import numpy as np
import datetime

# Día de hoy
hoy = pd.Timestamp.today().normalize()
# Último día de este mes
fin_mes_actual = hoy + pd.offsets.MonthEnd(0)
# Diferencia de días hasta fin de mes
dias_a_restar = (fin_mes_actual - hoy).days

# Configuración de página
st.set_page_config(
    page_title="Modelo USD Blue | Análisis Económico",
    layout="wide",
    page_icon="📈"
)
st.title("📈 Visualización del Modelo Econométrico del USD Blue")

# Descripción introductoria
st.markdown("""
Este dashboard interactivo permite visualizar el comportamiento histórico del dólar blue en Argentina,  así como las proyecciones con espectativas de corto y largo plazo generadas mediante un modelo econométrico autoregresivo.  
Este es un modelo one month ahead (predicción a un mes), por lo que las proyecciones para períodos t+2  
en adelante se realizan bajo el supuesto *ceteris paribus* en las variables macroeconómicas, es decir,  considerando que estas se mantienen constantes.    
Para la predicción diaria, se utiliza un modelo, Random Forest Regressor, optimizando sus hiperparámetros mediante búsqueda aleatoria con validación temporal (TimeSeriesSplit).    
Una vez ajustado el modelo, calcula el error absoluto medio (MAE) in-sample y sobre los últimos 30 días, para autoajustarse.

---
""")

# ID Google Sheets
SHEET_ID = "1jmzjQvTRWu9Loq_Gpn2SFCvVgo_qPo1X"
HOJAS = ["Datos Originales", "Prediccion mirada CP", "Prediccion mirada LP", "Real vs Predicho"]

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
    elif sheet_name in ["Prediccion mirada CP", "Prediccion mirada LP"]:
        df['Mes'] = pd.to_datetime(df['Mes'], errors='coerce')
        df = df.sort_values('Mes')
    elif sheet_name == "Real vs Predicho":
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df = df.sort_values('Fecha')
    return df

# ID Google Sheets para datos diarios
SHEET_ID_DIARIA = "1mCCiSDOdbp2lm90nnAAeQ9dBRO3Mh8_v"
HOJAS_DIARIAS = ["Prediccion Diaria vs Real Últimos 30 días", "Prediccion Diaria vs Real Historica"]

# Función para cargar hoja diaria
# @st.cache_data(show_spinner=True)
def cargar_hoja_diaria(sheet_id, sheet_name):
    sheet_name_encoded = urllib.parse.quote(sheet_name)
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name_encoded}"
    df = pd.read_csv(url)

    # Si existe columna "Fecha", la convertimos y ordenamos.
    if 'Fecha' in df.columns:
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df = df.sort_values('Fecha')
    return df

# Combinar HOJAS y HOJAS_DIARIAS
HOJAS_COMBINADAS = HOJAS + HOJAS_DIARIAS

# Sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/SantiagoWickham/ArgentinaInformalUSDPrediction/main/logo.jpg", width=100)
    st.header("⚙️ Configuración")
    
    # Selección hoja
    hoja_sel = st.selectbox("Seleccioná el tipo de gráfico", HOJAS_COMBINADAS)
    
    # Modo oscuro/claro
    modo_oscuro = st.checkbox("Modo oscuro", value=False)
    
    st.markdown("---")
    if hoja_sel == "Prediccion Diaria vs Real Últimos 30 días":
        st.markdown("📊 [Fuente de datos](https://docs.google.com/spreadsheets/d/1mCCiSDOdbp2lm90nnAAeQ9dBRO3Mh8_v)")
    else:
        st.markdown("📊 [Fuente de datos](https://docs.google.com/spreadsheets/d/1jmzjQvTRWu9Loq_Gpn2SFCvVgo_qPo1X)")

    # Opción para mostrar errores (residuos) solo para "Real vs Predicho"
    mostrar_residuos = False
    if hoja_sel == "Real vs Predicho":
        mostrar_residuos = st.checkbox("Mostrar errores de predicción (residuos)", value=False)
    # Opción para mostrar errores (residuos) solo para "Prediccion Diaria vs Real Últimos 30 días"
    if hoja_sel == "Prediccion Diaria vs Real Últimos 30 días":
        mostrar_residuos = st.checkbox("Mostrar errores de predicción diaria (residuos)", value=False)
    if hoja_sel == "Prediccion Diaria vs Real Historica":
        mostrar_residuos = st.checkbox("Mostrar errores de predicción diaria (residuos)", value=False)
# Carga datos según la hoja seleccionada
if hoja_sel in HOJAS:
    # Carga cualquiera de las hojas de SHEET_ID original
    df = cargar_hoja(SHEET_ID, hoja_sel)
elif hoja_sel in HOJAS_DIARIAS:
    # Carga la(s) hoja(s) de SHEET_ID_DIARIA
    df = cargar_hoja_diaria(SHEET_ID_DIARIA, hoja_sel)

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
    df_hist['MES'] = df_hist['MES'] + pd.offsets.MonthEnd(0) - pd.Timedelta(days=dias_a_restar)

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

elif hoja_sel == "Prediccion mirada CP":
    df_hist = cargar_hoja(SHEET_ID, "Datos Originales")
    fecha_6m_antes = df_hist['MES'].max() - pd.DateOffset(months=6)
    df_hist_cp = df_hist[df_hist['MES'] >= fecha_6m_antes]
    df_hist_cp['MES'] = df_hist_cp['MES'] + pd.offsets.MonthEnd(0) - pd.Timedelta(days=dias_a_restar)

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
        x=pd.concat([df['Mes'], df['Mes'][::-1]]),
        y=pd.concat([df['IC_Bajo_CP'], df['IC_Alto_CP'][::-1]]),
        fill='toself',
        fillcolor=COLOR_PALETA["intervalo_confianza"],
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=True,
        name='IC 95%'
    ))

    fig.update_layout(**layout_template("Predicción a Corto Plazo", modo_oscuro))

elif hoja_sel == "Prediccion mirada LP":
    df_hist = cargar_hoja(SHEET_ID, "Datos Originales")
    df_hist_lp = df_hist[df_hist['MES'] >= '2020-01-01']
    df_hist_lp['MES'] = df_hist_lp['MES'] + pd.offsets.MonthEnd(0) - pd.Timedelta(days=dias_a_restar)

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
            **layout_template("Real vs Predicho Mensual con errores", modo_oscuro),
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
        fig.update_layout(**layout_template("Real vs Predicho Mensual", modo_oscuro))

elif hoja_sel == "Prediccion Diaria vs Real Últimos 30 días":
    df_extra = cargar_hoja_diaria(SHEET_ID_DIARIA, "Prediccion Diaria vs Real Últimos 30 días")

    # Prediccion vs Real Últimos 30 días
    fig.add_trace(go.Scatter(
        x=df_extra['Fecha'],
        y=df_extra['Real'],
        mode='lines+markers',
        name='Prediccion Diaria vs Real Últimos 30 días',
        line=dict(color=COLOR_PALETA["real"], width=3),
        marker=dict(size=6),
        hovertemplate='%{x|%Y-%m-%d}: %{y:.2f} ARS<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=df_extra['Fecha'],
        y=df_extra['Predicción'],
        mode='lines+markers',
        name='USD Predicho (Extra)',
        line=dict(color=COLOR_PALETA["predicho_cp"], width=3, dash='dash'),
        marker=dict(size=6),
        hovertemplate='%{x|%Y-%m-%d}: %{y:.2f} ARS<extra></extra>'
    ))

    # Errores (residuos)
    if mostrar_residuos:
        residuos = df_extra['Real'] - df_extra['Predicción']
        fig.add_trace(go.Bar(
            x=df_extra['Fecha'],
            y=residuos,
            name='Error de predicción (residuo)',
            marker_color=COLOR_PALETA["error"],
            opacity=0.6,
            yaxis='y2',
            hovertemplate='%{x|%Y-%m-%d}: %{y:.2f} ARS<extra></extra>'
        ))
        fig.update_layout(
            **layout_template("Prediccion vs Real Últimos 30 días con errores", modo_oscuro),
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
        fig.update_layout(**layout_template("Prediccion vs Real Últimos 30 días", modo_oscuro))
        
elif hoja_sel == "Prediccion Diaria vs Real Historica":
    df_extra = cargar_hoja_diaria(SHEET_ID_DIARIA, "Prediccion Diaria vs Real Historica")
    fig.add_trace(go.Scatter(
        x=df_extra['Fecha'],
        y=df_extra['Real'],
        mode='lines+markers',
        name='Real (USD/ARS Blue)',
        line=dict(color=COLOR_PALETA["real"], width=0.5),
        marker=dict(size=3),
        hovertemplate='%{x|%Y-%m-%d}: %{y:.2f} ARS<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=df_extra['Fecha'],
        y=df_extra['Predicción'],
        mode='lines+markers',
        name='Predicción Diaria',
        line=dict(color=COLOR_PALETA["predicho_cp"], width=0.5, dash='dash'),
        marker=dict(size=3),
        hovertemplate='%{x|%Y-%m-%d}: %{y:.2f} ARS<extra></extra>'
    ))
    if mostrar_residuos:
        residuos = df_extra['Real'] - df_extra['Predicción']
        fig.add_trace(go.Bar(
            x=df_extra['Fecha'],
            y=residuos,
            name='Error de predicción (residuo)',
            marker_color=COLOR_PALETA["error"],
            opacity=0.6,
            yaxis='y2',
            hovertemplate='%{x|%Y-%m-%d}: %{y:.2f} ARS<extra></extra>'
        ))
        fig.update_layout(
            **layout_template("Predicción Diaria vs Real Histórica con errores", modo_oscuro),
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
        # Título corregido para cuando NO se muestran residuos
        fig.update_layout(**layout_template("Predicción Diaria vs Real Histórica", modo_oscuro))
        
# Mostrar gráfico
st.plotly_chart(fig, use_container_width=True)

# Si la hoja seleccionada es la diaria, cargamos y mostramos la tabla "Resumen"
if hoja_sel == "Prediccion Diaria vs Real Últimos 30 días":
    # 1) Cargamos la hoja "Resumen"
    df_resumen = cargar_hoja_diaria(SHEET_ID_DIARIA, "Resumen")

    # 2) Extraemos los dos valores que nos interesan
    mae_val = None
    prediccion_val = None

    # Recorremos las filas y guardamos valores según la descripción
    for idx, row in df_resumen.iterrows():
        desc = row["Descripción"]
        val = row["Valor"]
        if desc == "MAE últimos 30 días":
            mae_val = f"{float(val):.4f}"  # formateamos con cuatro decimales
        elif desc.startswith("Predicción para mañana"):
            prediccion_val = f"{float(val):.2f} ARS"  # dos decimales + unidad

    # 3) Mostramos un encabezado
    st.markdown("### Resumen de la predicción diaria")

    # 4) Creamos dos columnas y en cada una un st.metric
    col1, col2 = st.columns(2, gap="large")
    col1.metric(label="MAE últimos 30 días", value=mae_val)
    col2.metric(label="Predicción mañana (USD/ARS Blue)", value=prediccion_val)

# Botones de descarga CSV y PNG
import io
csv_buffer = df.to_csv(index=False).encode("utf-8")
img_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)

with st.sidebar:
    st.download_button("⬇️ Descargar CSV", data=csv_buffer, file_name=f"{hoja_sel.replace(' ', '_')}.csv", mime="text/csv")
    st.download_button("⬇️ Descargar imagen PNG", data=img_bytes, file_name=f"{hoja_sel.replace(' ', '_')}.png", mime="image/png")

# Sección colapsable "Sobre el modelo Mensual"
with st.expander("📖 Sobre el modelo Mensual"):
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
# Sección colapsable "Sobre el modelo Diario"
with st.expander("📖 Sobre el modelo Diario"):
    st.title("Metodología de Predicción del USD Blue")

    st.markdown("""
    **Metodología de Pronóstico Diario del USD Blue con Enfoque Rolling**

    ---

    ### 1. Propósito y Enfoque
    Desarrollar un sistema de pronóstico **one‑day‑ahead** para el precio del dólar paralelo en Argentina, basado en la evolución reciente de indicadores macroeconómicos clave. Se busca capturar patrones dinámicos y no lineales mediante un esquema de validación temporal iterativa (“rolling forecast”), asegurando que cada predicción utilice únicamente información disponible hasta ese día.

    ---

    ### 2. Componentes Fundamentales

    - **Selección de Variables**  
      - Indicadores macro exógenos: IPC, Reservas Internacionales, BADLAR, Riesgo País, Tipo de Cambio Oficial y Dólar MEP.  
      - Se asume que estas variables no son afectadas por el USD Blue en horizontes de 1–3 días.

    - **Construcción de Rezagos (Lags)**  
      - Para cada indicador, se incorporan sus valores de 1, 2 y 3 días atrás.  
      - Esto permite al modelo captar tanto efectos inmediatos como dinámicas de corto plazo.

    - **Modelo de Predicción**  
      - **Random Forest Regressor**: un ensamble de árboles que maneja relaciones no lineales y tolera outliers, sin requerir transformaciones manuales.

    ---

    ### 3. Validación Temporal (“Rolling Forecast”)

    1. **Partición Dinámica**  
       - En cada iteración, se entrena con datos hasta el día _t–1_ y se predice el día _t_, evitando filtraciones de información futura.

    2. **Reentrenamiento Continuo**  
       - Cada predicción utiliza un modelo reentrenado desde cero, garantizando el uso de la información más reciente.

    3. **Evaluación de Desempeño**  
       - **MAE histórico**: error absoluto medio sobre todas las predicciones fuera de muestra.  
       - **MAE últimos 30 días**: foco en el rendimiento reciente.

    ---

    ### 4. Optimización de Hiperparámetros

    - Búsqueda aleatoria con `RandomizedSearchCV` y `TimeSeriesSplit` (5 folds), ajustando:
      - `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` y `max_features`.  
    - Métrica objetivo: minimizar el **Mean Absolute Error (MAE)**.

    ---

    ### 5. Pronóstico en Producción

    1. Extraer los tres últimos valores reales de cada indicador.  
    2. Reentrenar el modelo con todo el histórico disponible.  
    3. Generar la predicción para el día siguiente (“one‑step‑ahead”).

    ---

    ### 6. Supuestos y Consideraciones

    - **Exogeneidad**: los indicadores macro no reaccionan al USD Blue en el corto plazo.  
    - **Estabilidad temporal**: las relaciones captadas por los lags perduran al menos 1–3 días.  
    - **Reentrenamiento frecuente**: mitiga la deriva del modelo, aunque incrementa coste computacional.

    ---

    Con este pipeline, garantizamos un **flujo end‑to‑end**: desde la ingesta y limpieza de datos, generación de rezagos, optimización y validación rolling, hasta el pronóstico en producción y la generación de reportes automatizados.
    """)

    
# Footer
st.markdown("---")
st.markdown("© 2025 Santiago Wickham | Proyecto económico - Datos: Fuentes oficiales: INDEC, BCRA")
st.markdown(""" 

🔗 [LinkedIn](https://www.linkedin.com/in/santiagowickham/)  
🐙 [GitHub](https://github.com/SantiagoWickham)

""")
