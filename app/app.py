import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(page_title="Demanda de Taxis NYC", layout="wide")

st.title("🚖 Pronóstico y Segmentación de Demanda de Taxis en NYC")
st.markdown("Interfaz interactiva conectada al modelo de Machine Learning del proyecto académico.")

# Rutas a los datos locales dentro de app/data/
DATA_DIR = "app/data"

# Cargar metadatos si existen
@st.cache_data
def cargar_datos():
    zonas_df = pd.read_csv(os.path.join(DATA_DIR, "zonas_segmentadas.csv"))
    with open(os.path.join(DATA_DIR, "info_modelo.json"), "r", encoding="utf-8") as f:
        info = json.load(f)
    return zonas_df, info

try:
    zonas_df, info = cargar_datos()
    st.success("✅ ¡Datos y metadatos cargados correctamente en Streamlit!")
    
    # Mostrar métricas del modelo obtenidas del json
    st.subheader("📊 Métricas de Desempeño del Modelo XGBoost")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE del Modelo (p50)", f"{info['metricas']['mae_modelo']:.2f}")
    col2.metric("MAE Línea Base", f"{info['metricas']['mae_baseline']:.2f}")
    col3.metric("Cobertura de Banda [p10, p90]", f"{info['metricas']['cobertura_banda']:.1f}%")

    st.markdown("---")
    st.subheader("🗺️ Exploración de Zonas Segmentadas")
    st.dataframe(zonas_df[["nombre_zona", "distrito", "nombre_grupo", "viajes_dia", "tarifa_mediana"]])

except Exception as e:
    st.error(f"⚠️ Asegúrate de tener la estructura de carpetas correcta (`app/app.py` y `app/data/...`). Error: {e}")