from fastapi import FastAPI, File, UploadFile
import pandas as pd
import io
import logging

# Importar las funciones de regresion y clasificación desde el módulo
from clasificacion import clasificacion_XGBoost
from regresion import regresion_energiaEntrada


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the classification API. Use /uploadfile/{model_type} to upload a CSV file."}

@app.post("/uploadfile/{model_type}")
async def TipoParticula(file: UploadFile):
    # Leer el archivo subido
    contents = await file.read()

    # Leer el contenido en un DataFrame de pandas
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

    predictions = clasificacion_XGBoost(df)
        
    # Convertir las predicciones a un formato adecuado para la respuesta
    response = ["It's a kaon" if pred == 1 else "It's a pion" for pred in predictions]
        
    return {"predictions": response}


@app.post("/uploadfile/{model_type}")
async def EnergiaEntrada(file: UploadFile):
    #Leer el archivo subido
    contents = await file.read()

    #Leer el contenido en un DataFrame de pandas
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

    predictions = EnergiaEntrada(df)

    return {"predicctions": predictions}
