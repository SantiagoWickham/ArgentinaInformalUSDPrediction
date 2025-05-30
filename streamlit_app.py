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

# Configuración Streamlit
st.set_page_config(page_title="Modelo de Predicción USD Blue", layout="wide")
st.title("📈 Modelo Econométrico para Predicción del Dólar Blue en Argentina")

financial_palette = ['#003f5c', '#2f4b7c', '#2f7c5e', '#7bcf6f', '#555555']
sns.set_palette(financial_palette)
sns.set_style("whitegrid")
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=financial_palette)

@st.cache_data(show_spinner=True)
def cargar_datos(sheet_id, sheet_name):
    # URL pública ejemplo con datos consolidados (cambia si tienes otro Google Sheet)
    url = f"https://docs.google.com/spreadsheets/d/e/2PACX-1vRAXkmSc6If8DaPCGgDX3GfhlvInDlajIUIHAztGwZGcdTa6k3SNRq2jhKthYOnNLQAFEb6_t2XPw1Y/pub?output=csv"
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
    df = df.copy()
    
    # Crear lags de variables
    df['RESERVAS_lag1'] = df['RESERVAS'].shift(1)
    df['BADLAR_lag1'] = df['BADLAR'].shift(1)
    
    df = df.dropna()

    X = df[['IPC', 'RESERVAS_lag1', 'BADLAR_lag1', 'RP', 'MEP']]
    y = df['USD_VENTA']

    X_const = sm.add_constant(X)

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X_const.iloc[:split_idx], X_const.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = sm.OLS(y_train, X_train).fit()
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    df_model = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})

    return model, df_model, X_test, y_test, y_pred, mae, rmse

def mostrar_resumen_regresion(model):
    st.subheader("📋 Resumen del Modelo")
    st.text(model.summary())

def mostrar_tests_diagnosticos(model):
    st.subheader("🔍 Tests Diagnósticos")
    resid = model.resid
    exog = model.model.exog

    # Breusch-Pagan (heterocedasticidad)
    bp_test = het_breuschpagan(resid, exog)
    st.write(f"Breusch-Pagan test p-value: {bp_test[1]:.4f}")

    # Durbin-Watson (autocorrelación)
    dw = durbin_watson(resid)
    st.write(f"Durbin-Watson statistic: {dw:.4f}")

    # Breusch-Godfrey (autocorrelación serial)
    bg_test = acorr_breusch_godfrey(model, nlags=2)
    st.write(f"Breusch-Godfrey test p-value: {bg_test[1]:.4f}")

def mostrar_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    st.write("### VIF (Factor de Inflación de la Varianza)")
    st.dataframe(vif_data)

def graficar_prediccion(y_test, y_pred):
    st.subheader("📈 Predicción vs Real")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_test.index, y_test, label="Real", marker='o')
    ax.plot(y_test.index, y_pred, label="Predicho", marker='x')
    ax.legend()
    ax.set_xlabel("Fecha")
    ax.set_ylabel("USD Blue")
    st.pyplot(fig)

def proyectar_dolar(df_model, model, meses=12, alpha=0.1):
    # Proyección simple usando último dato + coeficientes (sin input nuevo, para ejemplo)
    st.subheader(f"🔮 Proyección a {meses} meses")

    last_row = df_model.iloc[-1].copy()
    last_date = df_model.index[-1] if isinstance(df_model.index, pd.DatetimeIndex) else pd.Timestamp.today()

    coef = model.params
    residual_std = np.std(model.resid)

    fechas_futuras = [last_date + relativedelta(months=i) for i in range(1, meses+1)]
    predicciones = []

    for i, fecha in enumerate(fechas_futuras):
        # Aquí asumo que las variables independientes permanecen constantes (puedes mejorar)
        X_new = np.array([1,  # constante
                          last_row.get('IPC', np.nan),
                          last_row.get('RESERVAS_lag1', np.nan),
                          last_row.get('BADLAR_lag1', np.nan),
                          last_row.get('RP', np.nan),
                          last_row.get('MEP', np.nan)
                         ])
        pred = np.dot(coef.values, X_new)
        predicciones.append(pred)

    df_pred = pd.DataFrame({
        'Fecha': fechas_futuras,
        'Predicción USD Blue': predicciones
    })
    df_pred.set_index('Fecha', inplace=True)
    st.line_chart(df_pred)

    return df_pred

def mostrar_proyeccion(df_pred):
    st.subheader("📅 Tabla de Proyección")
    st.dataframe(df_pred)

# --- INTERFAZ ---

st.sidebar.header("Configuración")

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
mostrar_vif(X_test)

graficar_prediccion(y_test, y_pred)

pred_summary = proyectar_dolar(df_final, model, meses=meses_proyectar, alpha=alpha_conf)
mostrar_proyeccion(pred_summary)
