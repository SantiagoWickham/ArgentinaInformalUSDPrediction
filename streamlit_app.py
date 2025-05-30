import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.relativedelta import relativedelta

# Config Streamlit
st.set_page_config(page_title="Modelo de Predicción USD Blue", layout="wide")
st.title("📈 Modelo Econométrico para Predicción del Dólar Blue en Argentina")

financial_palette = ['#003f5c', '#2f4b7c', '#2f7c5e', '#7bcf6f', '#555555']
sns.set_palette(financial_palette)
sns.set_style("whitegrid")
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=financial_palette)

@st.cache_data(show_spinner=True)
def cargar_datos(sheet_id, sheet_name):
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    df = pd.read_csv(url)
    fechas = ['FECHA_USD', 'FECHA_IPC', 'FECHA_RP', 'FECHA_RESERVAS', 'FECHA_M2', 'FECHA_BADLAR', 'FECHA_TC', 'FECHA_MEP']
    for col in fechas:
        df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

    def fix_number_format(s):
        if pd.isna(s):
            return s
        s = str(s).replace(',', '.')
        if s.count('.') > 1:
            parts = s.split('.')
            s = ''.join(parts[:-1]) + '.' + parts[-1]
        return s

    variables = ['USD_VENTA', 'IPC', 'RP', 'RESERVAS', 'M2', 'BADLAR', 'TC', 'MEP']
    for col in variables:
        df[col] = df[col].apply(fix_number_format)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    def consolidar_mensual(fecha_col, valor_col):
        temp = df[[fecha_col, valor_col]].dropna()
        temp['MES'] = temp[fecha_col].dt.to_period('M')
        mensual = temp.sort_values(fecha_col).groupby('MES').last().reset_index()
        return mensual[['MES', valor_col]]

    usd = consolidar_mensual('FECHA_USD', 'USD_VENTA')
    ipc = consolidar_mensual('FECHA_IPC', 'IPC')
    rp = consolidar_mensual('FECHA_RP', 'RP')
    reservas = consolidar_mensual('FECHA_RESERVAS', 'RESERVAS')
    m2 = consolidar_mensual('FECHA_M2', 'M2')
    badlar = consolidar_mensual('FECHA_BADLAR', 'BADLAR')
    tc = consolidar_mensual('FECHA_TC', 'TC')
    mep = consolidar_mensual('FECHA_MEP', 'MEP')

    dfs = [usd, ipc, rp, reservas, m2, badlar, tc, mep]
    df_final = reduce(lambda left, right: pd.merge(left, right, on='MES', how='outer'), dfs)
    df_final['MES'] = df_final['MES'].dt.to_timestamp(how='end').dt.normalize()
    df_final = df_final.sort_values('MES').reset_index(drop=True)
    df_final = df_final.fillna(method='ffill')
    df_final.set_index('MES', inplace=True)

    return df_final

# El resto de tus funciones queda igual (ajustar_modelo_regresion, mostrar_resumen_regresion, etc.)

# --- INTERFAZ ---

st.sidebar.header("Configuración")

# Aquí pongo un Google Sheet público ejemplo con datos dummy, reemplazá por el tuyo
example_sheet_id = "1QXXrVZ_qZGQosMREdKdSqGz0Lq3V-t_gIq-BMFoI-Rc"  
example_sheet_name = "Sheet1"

sheet_id = st.sidebar.text_input("ID de Google Sheet", value=example_sheet_id)
sheet_name = st.sidebar.text_input("Nombre de la hoja", value=example_sheet_name)

meses_proyectar = st.sidebar.slider("Meses a proyectar", 1, 36, 12)
alpha_conf = st.sidebar.slider("Nivel de significancia para IC", 0.01, 0.3, 0.1, 0.01)

if not sheet_id.strip() or not sheet_name.strip():
    st.warning("Por favor, ingresa el ID de Google Sheet y el nombre de la hoja.")
    st.stop()

with st.spinner("Cargando y procesando datos..."):
    df_final = cargar_datos(sheet_id, sheet_name)

st.subheader("📅 Datos mensuales consolidados")
st.dataframe(df_final.tail(10))

model, df_model, X_test, y_test, y_pred, mae, rmse = ajustar_modelo_regresion(df_final)

mostrar_resumen_regresion(model)
mostrar_tests_diagnosticos(model)
mostrar_vif(model.model.exog)

graficar_prediccion(y_test, y_pred)
tabla_errores_por_test_size(df_model, sm.add_constant(df_model[['IPC', 'RESERVAS_lag1', 'BADLAR_lag1', 'RP', 'MEP']]), df_model['USD_VENTA'])

pred_summary = proyectar_dolar(df_model, model, meses=meses_proyectar, alpha=alpha_conf)
mostrar_proyeccion(pred_summary)

st.markdown("""
---
_App creada con Streamlit por ChatGPT para análisis y predicción económica._  
Puedes ajustar las variables y extender el modelo para mejorar la proyección.
""")
