import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import urllib.parse
import numpy as np

st.set_page_config(page_title="Gráficos USD Blue", layout="wide")
st.title("Visualización modelo econométrico USD Blue")

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

# Carga todas las hojas
hojas = ["Datos Originales", "Prediccion_CP", "Prediccion_LP", "Real vs Predicho"]
data = {hoja: cargar_hoja(sheet_id, hoja) for hoja in hojas}

hoja_sel = st.selectbox("Selecciona hoja para graficar:", hojas)

df = data[hoja_sel]
#st.write(f"Vista previa datos hoja **{hoja_sel}**")
#st.dataframe(df.head())

sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(12,6))

if hoja_sel == "Datos Originales":
    # Mostrar datos desde 2020-01-01
    df_hist = df[df['MES'] >= '2020-01-01']
    ax.plot(df_hist['MES'], df_hist['USD_VENTA'], label='USD Blue', color='#003f5c', linewidth=2)
    ax.set_title("Datos Originales: USD Blue (desde 2020)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio USD Blue (ARS)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    # Formato fechas
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
elif hoja_sel == "Prediccion_CP":
    # Últimos 6 meses de datos reales + predicción CP con IC
    df_hist = data["Datos Originales"]
    fecha_6m_antes = df_hist['MES'].max() - pd.DateOffset(months=6)
    df_hist_cp = df_hist[df_hist['MES'] >= fecha_6m_antes]

    # Datos para conectar último dato real con primer IC
    ultimo_real_fecha = df_hist_cp['MES'].max()
    ultimo_real_valor = df_hist_cp.loc[df_hist_cp['MES'] == ultimo_real_fecha, 'USD_VENTA'].values[0]

    primera_pred_fecha = df['Mes'].min()
    ic_bajo_1 = df.loc[df['Mes'] == primera_pred_fecha, 'IC_Bajo_CP'].values[0]
    ic_alto_1 = df.loc[df['Mes'] == primera_pred_fecha, 'IC_Alto_CP'].values[0]

    # Convertir fechas a números para matplotlib
    ultimo_real_fecha_num = mdates.date2num(ultimo_real_fecha)
    primera_pred_fecha_num = mdates.date2num(primera_pred_fecha)

    # Graficar últimos datos reales
    ax.plot(df_hist_cp['MES'], df_hist_cp['USD_VENTA'], label='USD Real', color='#2f4b7c', linewidth=2)

    # Graficar línea predicción CP
    ax.plot(df['Mes'], df['USD_Predicho_CP'], label='Predicción CP', color='#2f7c5e', linewidth=2, linestyle='--')

    # Dibujar las líneas que conectan último punto real con IC bajo y alto del primer mes predicho
    ax.plot([ultimo_real_fecha_num, primera_pred_fecha_num],
            [ultimo_real_valor, ic_bajo_1],
            color='#2f7c5e', linewidth=1.5)
    ax.plot([ultimo_real_fecha_num, primera_pred_fecha_num],
            [ultimo_real_valor, ic_alto_1],
            color='#2f7c5e', linewidth=1.5)

    # Rellenar área triangular entre esas dos líneas para suavizar el inicio del IC
    ax.fill_between([ultimo_real_fecha_num, primera_pred_fecha_num],
                    [ultimo_real_valor, ic_bajo_1],
                    [ultimo_real_valor, ic_alto_1],
                    color='#2f7c5e', alpha=0.25)

    # Luego rellena normalmente el IC desde la predicción (mes 1 en adelante)
    fechas_pred_num = mdates.date2num(df['Mes'])
    ax.fill_between(fechas_pred_num, df['IC_Bajo_CP'], df['IC_Alto_CP'], color='#2f7c5e', alpha=0.25, label='IC 95%')

    ax.set_title("Predicción Corto Plazo (últimos 6 meses reales + predicción)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio USD Blue (ARS)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

elif hoja_sel == "Prediccion_LP":
    # Mostrar datos desde 2020 + predicción LP con IC
    df_hist = data["Datos Originales"]
    df_hist_lp = df_hist[df_hist['MES'] >= '2020-01-01']
    
    ax.plot(df_hist_lp['MES'], df_hist_lp['USD_VENTA'], label='USD Real', color='#003f5c', linewidth=2)
    ax.plot(df['Mes'], df['USD_Predicho_LP'], label='Predicción LP', color='#7bcf6f', linewidth=2, linestyle='--')
    ax.fill_between(df['Mes'], df['IC_Bajo_LP'], df['IC_Alto_LP'], color='#7bcf6f', alpha=0.25, label='IC 95%')
    
    ax.set_title("Predicción Largo Plazo (desde 2020)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio USD Blue (ARS)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

elif hoja_sel == "Real vs Predicho":
    ax.plot(df['Fecha'], df['USD_Real'], label='USD Real', color='black', linewidth=2)
    ax.plot(df['Fecha'], df['USD_Predicho'], label='USD Predicho', color='red', linewidth=2, linestyle='--')
    ax.set_title("Real vs Predicho")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio USD Blue (ARS)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

st.pyplot(fig)
