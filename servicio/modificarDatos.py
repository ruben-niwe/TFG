import pandas as pd
import numpy as np


def modificar_datos(df):
    df['hitX'] = df['hitX'] + 100
    df['hitZ'] =df['hitZ'] - 150
    return df

def filtrado_datos_energia(df, N):
    kaones = []
    labels = []
    valores_trueE = []  # Usamos un conjunto para almacenar valores únicos de trueE

    df_sorted = df.sort_values(by=['eventID', 'hitTime'], ascending=[True, False])

    for eventID, grupo in df_sorted.groupby('eventID'):
        pdgCodes = grupo['PDGcode'].unique()

        for pdgCode in pdgCodes:
            grupo_filtrado = grupo[grupo['PDGcode'] == pdgCode]
            grupo_ordenado = grupo_filtrado.head(N)

            # Obtener el valor de 'trueE' para el evento actual
            trueE_value = grupo_filtrado['trueE'].iloc[0]

            # Almacenar el valor de trueE en el conjunto
            valores_trueE.append(trueE_value)

            # Inicializar arrays para el padding
            hitX_padded = np.zeros(N)
            hitY_padded = np.zeros(N)
            hitZ_padded = np.zeros(N)
            hitInteg_padded = np.zeros(N)

            # Separar y aplicar padding a los valores de hitX, hitY, hitZ, hitInteg
            hitX_padded[:len(grupo_ordenado['hitX'])] = grupo_ordenado['hitX']
            hitY_padded[:len(grupo_ordenado['hitY'])] = grupo_ordenado['hitY']
            hitZ_padded[:len(grupo_ordenado['hitZ'])] = grupo_ordenado['hitZ']
            hitInteg_padded[:len(grupo_ordenado['hitInteg'])] = grupo_ordenado['hitInteg']

            # Concatenar los valores ya con el padding aplicado, añadiendo 'trueE' al principio
            hit_values_reorganized = np.concatenate([[trueE_value], hitX_padded, hitY_padded, hitZ_padded, hitInteg_padded])

            kaones.append(hit_values_reorganized)

            # Modificar las etiquetas de 211 a 0 y de 321 a 1
            if pdgCode == 211:
                labels.append(0)
            elif pdgCode == 321:
                labels.append(1)

    return np.array(kaones), np.array(labels), np.array(list(valores_trueE))


def filtrado_datos(df, N):
    kaones = []
    labels = []

    df_sorted = df.sort_values(by=['eventID', 'hitTime'], ascending=[True, False])

    for eventID, grupo in df_sorted.groupby('eventID'):
        pdgCodes = grupo['PDGcode'].unique()

        for pdgCode in pdgCodes:
            grupo_filtrado = grupo[grupo['PDGcode'] == pdgCode]
            grupo_ordenado = grupo_filtrado.head(N)

            # Inicializar arrays para el padding
            hitX_padded = np.zeros(N)
            hitY_padded = np.zeros(N)
            hitZ_padded = np.zeros(N)
            hitInteg_padded = np.zeros(N)

            # Separar y aplicar padding a los valores de hitX, hitY, hitZ, hitInteg
            hitX_padded[:len(grupo_ordenado['hitX'])] = grupo_ordenado['hitX']
            hitY_padded[:len(grupo_ordenado['hitY'])] = grupo_ordenado['hitY']
            hitZ_padded[:len(grupo_ordenado['hitZ'])] = grupo_ordenado['hitZ']
            hitInteg_padded[:len(grupo_ordenado['hitInteg'])] = grupo_ordenado['hitInteg']

            # Concatenar los valores ya con el padding aplicado
            hit_values_reorganized = np.concatenate([hitX_padded, hitY_padded, hitZ_padded, hitInteg_padded])

            kaones.append(hit_values_reorganized)

            # Modificar las etiquetas de 211 a 0 y de 321 a 1
            if pdgCode == 211:
                labels.append(0)
            elif pdgCode == 321:
                labels.append(1)

    return np.array(kaones), np.array(labels)



def filtrado_datos_energia_particula(df_train, N):
    X_train, etiqueta_particula_train, trueE_train = filtrado_datos_energia(df_train, N)    
    X_train = np.column_stack((etiqueta_particula_train, X_train[:, 1:]))
    return X_train, trueE_train



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