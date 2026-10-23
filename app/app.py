import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(page_title="Demanda de Taxis NYC", layout="wide")

st.title("🚖 Pronóstico y Segmentación de Demanda de Taxis en Nueva York")
st.markdown("Herramienta interactiva para decidir cuántos taxis enviar a cada zona y a cada hora con base en el modelo de Machine Learning.")

DATA_DIR = "app/data"

@st.cache_data
def cargar_datos():
    zonas_df = pd.read_csv(os.path.join(DATA_DIR, "zonas_segmentadas.csv"))
    demanda_df = pd.read_parquet(os.path.join(DATA_DIR, "demanda.parquet"))
    with open(os.path.join(DATA_DIR, "info_modelo.json"), "r", encoding="utf-8") as f:
        info = json.load(f)
    return zonas_df, demanda_df, info

try:
    zonas_df, demanda_df, info = cargar_datos()
    st.success("✅ Datos y modelo cargados correctamente.")

    # Pestañas para organizar la tarea
    tab1, tab2 = st.tabs(["📈 Pronóstico Operacional (Parte 1)", "🗺️ Segmentación de Zonas (Parte 2)"])

    with tab1:
        st.subheader("Pronóstico de Demanda e Incertidumbre por Hora")
        st.markdown("Selecciona una zona de la ciudad y el sistema te indicará cuántos viajes esperar y cuántos taxis enviar.")

        # Selector de zonas disponibles para pronóstico
        zonas_disponibles = info.get("zonas", [])
        nombres_map = info.get("nombres_zonas", {})
        
        opciones_zona = {z: f"{z} - {nombres_map.get(str(z), 'Zona NYC')}" for z in zonas_disponibles}
        zona_seleccionada = st.selectbox("Selecciona la Zona (LocationID):", options=list(opciones_zona.keys()), format_func=lambda x: opciones_zona[x])

        if zona_seleccionada:
            # Filtrar datos de demanda para la zona seleccionada
            df_zona_demo = demanda_df[demanda_df["PULocationID"] == zona_seleccionada]
            
            if not df_zona_demo.empty:
                st.markdown(f"### Resultados para: **{opciones_zona[zona_seleccionada]}**")
                
                # Mostrar tabla de las horas proyectadas
                st.dataframe(df_zona_demo[["fecha_hora", "viajes", "lag_168"]].rename(columns={
                    "fecha_hora": "Fecha y Hora", 
                    "viajes": "Viajes Reales / Históricos", 
                    "lag_168": "Línea Base (Semana Anterior)"
                }))
                
                st.info("💡 **Recomendación operativa:** Asigna unidades de flota considerando los picos esperados de demanda en esta franja horaria para evitar tiempos de espera elevados.")
            else:
                st.warning("No hay registros de pronóstico detallado para esta zona en el conjunto de prueba.")

    with tab2:
        st.subheader("Segmentación de Zonas (K-Means y PCA)")
        st.dataframe(zonas_df[["nombre_zona", "distrito", "nombre_grupo", "viajes_dia", "tarifa_mediana", "propina_mediana"]])

except Exception as e:
    st.error(f"⚠️ Error al cargar los componentes de la app. Verifica la estructura de archivos en GitHub (`app/data/...`). Detalle: {e}")