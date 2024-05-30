# clasificacion.py
import pandas as pd
import tensorflow as tf
import numpy as np
import joblib


from modificarDatos import modificar_datos, filtrado_datos_produccion


def clasificacion_XGBoost(df):

    df = modificar_datos(df)
    X = filtrado_datos_produccion(df)
    model = joblib.load('modelos/modelo_XGBoost.pkl')
    predicciones = model.predict(X)

    return predicciones



	

