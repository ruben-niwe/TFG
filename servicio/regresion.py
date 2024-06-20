# regresion.py
import pandas as pd
import joblib
import numpy as np
from clasificacion import clasificacion_XGBoost
from modificarDatos import filtrado_datos_produccion



def regresion_energiaEntrada(df):

    N = 800

    joblib_file = 'modelos/model_regression_Random_Forest.pkl'
    rf_loaded = joblib.load(joblib_file)
    predicciones_clasificacion = clasificacion_XGBoost(df)

    X_sin_particula = filtrado_datos_produccion(df, N)

    X = np.column_stack((predicciones_clasificacion, X_sin_particula))

    trueE_predecida = rf_loaded.predict(X)

    return predicciones_clasificacion, trueE_predecida