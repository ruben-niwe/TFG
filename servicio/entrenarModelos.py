import pandas as pd
import numpy as np
import os

import joblib

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier

from modificarDatos import filtrado_datos_energia_particula, filtrado_datos

def preparar_datos(N):
    path_actual = os.getcwd()
    subdirectorio = 'datas'
    file_train = 'df_train.csv'
    file_valid = 'df_valid.csv'

    path_train = os.path.join(path_actual, subdirectorio, file_train)
    path_valid = os.path.join(path_actual, subdirectorio, file_valid)

    df_train = pd.read_csv(path_train, index_col=False)
    df_valid = pd.read_csv(path_valid, index_col=False)
    df_union = pd.concat([df_train, df_valid], ignore_index=True)

    return df_union


def entrenarRegresion():
    print('Inicio entrenamiento regresion')
    N = 800

    df_union = preparar_datos(N)

    model_RF = RandomForestRegressor(n_estimators=50, max_depth=22, min_samples_split=2, min_samples_leaf=2, max_features='sqrt')

    X, trueE = filtrado_datos_energia_particula(df_union, N)

    model_RF.fit(X, trueE)

    joblib.dump(model_RF, 'modelos/model_regression_Random_Forest.pkl')
    print('FIn entrenamiento regresion')



def entrenarClasificacion():
    print('Inicio entrenamiento Clasificacion')
    N = 760

    df_union = preparar_datos(N)

    X, y = filtrado_datos(df_union, N)

    params = {
        'booster': 'gbtree',
        'max_depth': 6,
        'gamma': 0,
        'learning_rate': 0.3,
        'subsample': 1,
        'colsample_bytree': 1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': 42
    }

    num_epochs = 9
    model = XGBClassifier(**params, n_estimators=num_epochs)
    model.fit(X, y)
    joblib.dump(model, 'modelos/model_classification_XGBoost.pkl')
    print('Fin entrenamiento Clasificacion')



def main():
    entrenarClasificacion()
    entrenarRegresion()

if __name__ == "__main__":
    main()



