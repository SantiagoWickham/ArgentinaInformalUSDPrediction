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
import datetime as dt
from dateutil.relativedelta import relativedelta

# Config Streamlit
st.set_page_config(page_title="Modelo de Predicción USD Blue", layout="wide")
st.title("📈 Modelo Econométrico para Predicción del Dólar Blue en Argentina")

# Paleta financiera y estilo
financial_palette = ['#003f5c', '#2f4b7c', '#2f7c5e', '#7bcf6f', '#555555']
sns.set_palette(financial_palette)
sns.set_style("whitegrid")
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=financial_palette)

# --- Funciones ---

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

def ajustar_modelo_regresion(df):
    # Variables con lags y filtrado
    df['RESERVAS_lag1'] = df['RESERVAS'].shift(1)
    df['BADLAR_lag1'] = df['BADLAR'].shift(1)
    df['M2_lag1'] = df['M2'].shift(1)
    df = df[df.index >= '2020-03-01']
    df_model = df.dropna()

    y = df_model['USD_VENTA']
    X = df_model[['IPC', 'RESERVAS_lag1', 'BADLAR_lag1', 'RP', 'MEP']]
    X = sm.add_constant(X)

    # Train/Test split 85%
    train_size = int(len(df_model) * 0.85)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    model = sm.OLS(y_train, X_train).fit()
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return model, df_model, X_test, y_test, y_pred, mae, rmse

def mostrar_resumen_regresion(model):
    st.subheader("📊 Resumen del Modelo de Regresión")
    st.text(model.summary())

def mostrar_tests_diagnosticos(model):
    st.subheader("🧪 Tests Diagnósticos")
    resid = model.resid
    exog = model.model.exog

    bp_test = het_breuschpagan(resid, exog)
    dw_stat = durbin_watson(resid)
    bg_test = acorr_breusch_godfrey(model, nlags=1)

    st.write(f"**Breusch-Pagan p-value (heterocedasticidad):** {bp_test[1]:.4f}")
    st.write(f"**Durbin-Watson (autocorrelación residuos):** {dw_stat:.2f}")
    st.write(f"**Breusch-Godfrey p-value (autocorrelación):** {bg_test[1]:.4f}")

def mostrar_vif(X):
    st.subheader("📈 VIF (Factor de Inflación de la Varianza)")
    vif_data = pd.DataFrame()
    vif_data['Variable'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    st.dataframe(vif_data)

def graficar_prediccion(y_test, y_pred):
    st.subheader("📉 Comparación Real vs. Predicho")
    plt.figure(figsize=(14,5))
    plt.plot(y_test.index, y_test, label='Real', marker='o')
    plt.plot(y_test.index, y_pred, label='Predicho', marker='x')
    plt.title('Real vs. Predicho (Test)')
    plt.xlabel('Fecha')
    plt.ylabel('USD_VENTA')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    plt.clf()

def tabla_errores_por_test_size(df_model, X, y):
    st.subheader("📊 Errores MAE y RMSE según tamaño de Test")
    test_sizes = np.arange(0.1, 0.6, 0.05)
    results = []

    for test_size in test_sizes:
        train_size = int(len(df_model) * (1 - test_size))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        model = sm.OLS(y_train, X_train).fit()
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results.append({'Test Size': test_size, 'Train Size': 1-test_size, 'MAE': mae, 'RMSE': rmse})

    df_results = pd.DataFrame(results)
    st.dataframe(df_results.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}"}))

    # Gráficos
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(df_results['Test Size'], df_results['MAE'], marker='o')
    axs[0].set_title('MAE vs Tamaño Test')
    axs[0].set_xlabel('Proporción Test')
    axs[0].set_ylabel('MAE')
    axs[0].grid(True)

    axs[1].plot(df_results['Test Size'], df_results['RMSE'], marker='o', color=financial_palette[1])
    axs[1].set_title('RMSE vs Tamaño Test')
    axs[1].set_xlabel('Proporción Test')
    axs[1].set_ylabel('RMSE')
    axs[1].grid(True)

    st.pyplot(fig)
    plt.clf()

def proyectar_dolar(df_model, model, meses=12, alpha=0.1):
    ultimo_mes = df_model.index.max()
    fechas_futuras = [ultimo_mes + relativedelta(months=i) for i in range(1, meses+1)]

    # Últimos datos para variables predictoras
    ult_ipc = df_model['IPC'].iloc[-1]
    ult_rp = df_model['RP'].iloc[-1]
    ult_mep = df_model['MEP'].iloc[-1]
    ult_reservas = df_model['RESERVAS'].iloc[-1]
    ult_badlar = df_model['BADLAR'].iloc[-1]

    data_pred = []
    for mes in fechas_futuras:
        # Suponemos valores constantes para las variables exógenas (puedes mejorar)
        fila = {
            'const': 1,
            'IPC': ult_ipc,
            'RP': ult_rp,
            'MEP': ult_mep,
            'RESERVAS_lag1': ult_reservas,
            'BADLAR_lag1': ult_badlar
        }
        data_pred.append(fila)

    X_pred = pd.DataFrame(data_pred, index=fechas_futuras)
    pred = model.get_prediction(X_pred)
    pred_summary = pred.summary_frame(alpha=alpha)
    pred_summary['MES'] = fechas_futuras

    return pred_summary

def mostrar_proyeccion(pred_summary):
    st.subheader("🔮 Proyección del Dólar Blue")
    st.dataframe(pred_summary.style.format({
        'mean': '{:.2f}', 'mean_ci_lower': '{:.2f}', 'mean_ci_upper': '{:.2f}'
    }))

    plt.figure(figsize=(14,5))
    plt.plot(pred_summary['MES'], pred_summary['mean'], label='Predicción', marker='o')
    plt.fill_between(pred_summary['MES'], pred_summary['mean_ci_lower'], pred_summary['mean_ci_upper'], color='b', alpha=0.2, label='Intervalo Confianza')
    plt.title('Proyección del Dólar Blue con Intervalo de Confianza')
    plt.xlabel('Mes')
    plt.ylabel('USD_VENTA')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    plt.clf()

# --- INTERFAZ ---

st.sidebar.header("Configuración")
sheet_id = st.sidebar.text_input("ID de Google Sheet", value="TU_ID_DE_SHEET_AQUI")
sheet_name = st.sidebar.text_input("Nombre de la hoja", value="Sheet1")
meses_proyectar = st.sidebar.slider("Meses a proyectar", min_value=1, max_value=36, value=12)
alpha_conf = st.sidebar.slider("Nivel de significancia para IC", min_value=0.01, max_value=0.3, value=0.1, step=0.01)

if sheet_id.strip() == "" or sheet_name.strip() == "":
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
