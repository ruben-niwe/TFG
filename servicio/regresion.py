import pandas as pd
import joblib
import numpy as np
from clasificacion import clasificacion_XGBoost


joblib_file = "random_forest_model.pkl"
rf_loaded = joblib.load(joblib_file)
predictions = rf_loaded.predict(X_new)

def regresion_energiaEntrada(df):

    joblib_file = "random_forest_model.pkl"
    rf_loaded = joblib.load(joblib_file)
    predicciones = rf_loaded.predict(X)
     
    return predicciones