# clasificacion.py
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import joblib

def filtrado_datos_variable(df):
    kaones =[]

    df_sorted = df.sort_values(by=['eventID', 'hitTime'], ascending=[True, False])

    for eventID, grupo in df_sorted.groupby('eventID'):
        pdgCodes = grupo['PDGcode'].unique()

        for pdgCode in pdgCodes:
            grupo_filtrado = grupo[grupo['PDGcode']==pdgCode]

            # Extraer directamente los valores
            hitX_values = grupo_filtrado['hitX'].values
            hitY_values = grupo_filtrado['hitY'].values
            hitZ_values = grupo_filtrado['hitZ'].values

            hitInteg_values = grupo_filtrado['hitInteg'].values

            hit_vales_reorganized = np.concatenate([hitX_values, hitY_values, hitZ_values, hitInteg_values])
            kaones.append(hit_vales_reorganized)

    return kaones

def filtrado_datos(df,N=760):
    kaones = []
    df_sorted = df.sort_values(by=['eventID', 'hitTime'], ascending=[True, False])

    for eventID, grupo in df_sorted.groupby('eventID'):
        pdgCodes = grupo['PDGcode'].unique()

        for pdgCode in pdgCodes:
            grupo_filtrado = grupo[grupo['PDGcode']==pdgCode]
            grupo_ordenado = grupo_filtrado.head(N)

            #inicializar arrays para el padding
            hitX_padded = np.zeros(N)
            hitY_padded = np.zeros(N)
            hitZ_padded = np.zeros(N)
            hitInteg_padded = np.zeros(N)

            #Separa y aplicar padding a los valores de hitX, hitY, hitZ y hitInteg
            hitX_padded[:len(grupo_ordenado['hitX'])] = grupo_ordenado['hitX']
            hitY_padded[:len(grupo_ordenado['hitY'])] = grupo_ordenado['hitY']
            hitZ_padded[:len(grupo_ordenado['hitZ'])] = grupo_ordenado['hitZ']
            hitInteg_padded[:len(grupo_ordenado['hitInteg'])] = grupo_ordenado['hitInteg']

            #Concatenar los valores ya con el padding aplicado
            hit_values_reorganized = np.concatenate([hitX_padded, hitY_padded, hitZ_padded, hitInteg_padded])

            kaones.append(hit_values_reorganized)

    return np.array(kaones)


def filtrado_datos_produccion(df, N=760):
    kaones = []
    df_sorted = df.sort_values(by=['eventID', 'hitTime'], ascending=[True, False])

    for eventID, grupo in df_sorted.groupby('eventID'):
        grupo_ordenado = grupo.head(N)

        # Inicializar arrays para el padding
        hitX_padded = np.zeros(N)
        hitY_padded = np.zeros(N)
        hitZ_padded = np.zeros(N)
        hitInteg_padded = np.zeros(N)

        # Separar y aplicar padding a los valores de hitX, hitY, hitZ y hitInteg
        hitX_padded[:len(grupo_ordenado['hitX'])] = grupo_ordenado['hitX']
        hitY_padded[:len(grupo_ordenado['hitY'])] = grupo_ordenado['hitY']
        hitZ_padded[:len(grupo_ordenado['hitZ'])] = grupo_ordenado['hitZ']
        hitInteg_padded[:len(grupo_ordenado['hitInteg'])] = grupo_ordenado['hitInteg']

        # Concatenar los valores ya con el padding aplicado
        hit_values_reorganized = np.concatenate([hitX_padded, hitY_padded, hitZ_padded, hitInteg_padded])

        kaones.append(hit_values_reorganized)

    return np.array(kaones)




def clasificacion_XGBoost(df):

    X = filtrado_datos_produccion(df)
    model = joblib.load('modelos/modelo_XGBoost.pkl')
    predicciones = model.predict(X)

    return predicciones




"""
def clasificicacion_RNN(df):
    max_length = 6328

    X = filtrado_datos_variable(df)

    X_p = pad_sequences(X, maxlen=max_length, padding='post', dtype='float32')
    X_p = np.expand_dims(X_p, -1)

    model = load_model('modelos/modelo_CNN.h5')

    predicciones = model.predict(X_p)

    predicciones = (predicciones > 0.5).astype(int)
    return predicciones
"""

	

