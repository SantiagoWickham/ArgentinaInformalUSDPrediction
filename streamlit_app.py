import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Gráficos USD Blue", layout="wide")
st.title("Visualización modelo econométrico USD Blue")

sheet_id = "1jmzjQvTRWu9Loq_Gpn2SFCvVgo_qPo1X"

@st.cache_data(show_spinner=True)
def cargar_hoja(sheet_id, sheet_name):
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    df = pd.read_csv(url)
    # La columna con fecha varía según hoja:
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
st.write(f"Vista previa datos hoja **{hoja_sel}**")
st.dataframe(df.head())

plt.figure(figsize=(12,6))
sns.set_style("whitegrid")

if hoja_sel == "Datos Originales":
    # Grafico MES vs USD_VENTA
    plt.plot(df['MES'], df['USD_VENTA'], label='USD Venta')
    plt.title("Datos Originales: USD Venta")
    plt.xlabel("Mes")
    plt.ylabel("USD Venta")
    plt.legend()

elif hoja_sel == "Prediccion_CP":
    # Grafico Mes vs USD_Predicho_CP con bandas IC_Bajo_CP y IC_Alto_CP
    plt.plot(df['Mes'], df['USD_Predicho_CP'], label='Predicción CP', color='blue')
    plt.fill_between(df['Mes'], df['IC_Bajo_CP'], df['IC_Alto_CP'], color='blue', alpha=0.2, label='IC 95%')
    plt.title("Predicción Corto Plazo")
    plt.xlabel("Mes")
    plt.ylabel("USD Predicho CP")
    plt.legend()

elif hoja_sel == "Prediccion_LP":
    # Grafico Mes vs USD_Predicho_LP con bandas IC_Bajo_LP y IC_Alto_LP
    plt.plot(df['Mes'], df['USD_Predicho_LP'], label='Predicción LP', color='green')
    plt.fill_between(df['Mes'], df['IC_Bajo_LP'], df['IC_Alto_LP'], color='green', alpha=0.2, label='IC 95%')
    plt.title("Predicción Largo Plazo")
    plt.xlabel("Mes")
    plt.ylabel("USD Predicho LP")
    plt.legend()

elif hoja_sel == "Real vs Predicho":
    # Grafico Fecha vs USD_Real y USD_Predicho
    plt.plot(df['Fecha'], df['USD_Real'], label='USD Real', color='black')
    plt.plot(df['Fecha'], df['USD_Predicho'], label='USD Predicho', color='red')
    plt.title("Real vs Predicho")
    plt.xlabel("Fecha")
    plt.ylabel("USD")
    plt.legend()

plt.grid(True)
st.pyplot(plt.gcf())
