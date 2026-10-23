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

    tab1, tab2 = st.tabs(["📈 Pronóstico Operacional y Precisión", "🗺️ Segmentación de Zonas"])

    with tab1:
        st.subheader("Pronóstico de Demanda e Incertidumbre por Hora")
        zonas_disponibles = info.get("zonas", [])
        nombres_map = info.get("nombres_zonas", {})
        
        opciones_zona = {z: f"{z} - {nombres_map.get(str(z), 'Zona NYC')}" for z in zonas_disponibles}
        zona_seleccionada = st.selectbox("Selecciona la Zona (LocationID):", options=list(opciones_zona.keys()), format_func=lambda x: opciones_zona[x])

        if zona_seleccionada:
            df_zona_demo = demanda_df[demanda_df["PULocationID"] == zona_seleccionada]
            
            if not df_zona_demo.empty:
                st.markdown(f"### Análisis para: **{opciones_zona[zona_seleccionada]}**")
                
                # CREAMOS DOS COLUMNAS: Izquierda la tabla, Derecha la precisión y métricas
                col_tabla, col_precision = st.columns([2, 1])

                with col_tabla:
                    st.markdown("#### 📋 Tabla de Valores (Real vs Línea Base)")
                    st.dataframe(df_zona_demo[["fecha_hora", "viajes", "lag_168"]].rename(columns={
                        "fecha_hora": "Fecha y Hora", 
                        "viajes": "Viajes Registrados", 
                        "lag_168": "Línea Base (Semana Anterior)"
                    }), use_container_width=True)

                with col_precision:
                    st.markdown("#### 🎯 Precisión y Desempeño")
                    # Métraicas extraídas del JSON del modelo
                    mae_mod = info["metricas"]["mae_modelo"]
                    mae_base = info["metricas"]["mae_baseline"]
                    cobertura = info["metricas"]["cobertura_banda"]

                    st.metric(label="MAE del Modelo (p50)", value=f"{mae_mod:.2f}")
                    st.metric(label="MAE Línea Base (lag_168)", value=f"{mae_base:.2f}")
                    st.metric(label="Cobertura Banda [p10, p90]", value=f"{cobertura:.1f}%")

                    st.info(
                        "**Nota de Precisión:** "
                        "El modelo evalúa la calidad de las predicciones de demanda por cuantiles, "
                        "minimizando los errores operativos frente a los falsos positivos en la asignación de flotas."
                    )
                
                st.markdown("---")
                st.info("💡 **Recomendación operativa:** Asigna unidades de flota considerando los picos esperados de demanda en esta franja horaria.")
            else:
                st.warning("No hay registros detallados para esta zona.")

    with tab2:
        st.subheader("Segmentación de Zonas (K-Means y PCA)")
        st.dataframe(zonas_df[["nombre_zona", "distrito", "nombre_grupo", "viajes_dia", "tarifa_mediana", "propina_mediana"]], use_container_width=True)

except Exception as e:
    st.error(f"⚠️ Error al cargar los componentes. Detalle: {e}")