from fastapi import FastAPI
import xgboost as xgb
import pandas as pd
import json
import os
import numpy as np

app = FastAPI(title="API de Demanda de Taxis NYC", version="1.0")

# Cargar metadatos si existen
info_path = "info_modelo.json"
if os.path.exists(info_path):
    with open(info_path, "r", encoding="utf-8") as f:
        info_modelo = json.load(f)
else:
    info_modelo = {}

# Cargar el modelo XGBoost entrenado (.ubj)
model_path = "modelo_pronostico.ubj"
model = None
if os.path.exists(model_path):
    model = xgb.XGBRegressor()
    model.load_model(model_path)

@app.get("/")
def home():
    return {"mensaje": "¡Bienvenido a la API de pronóstico de demanda de taxis en NYC!"}

@app.get("/health")
def health_check():
    return {
        "estado": "ok",
        "modelo_cargado": model is not None,
        "zonas_disponibles": len(info_modelo.get("zonas", []))
    }