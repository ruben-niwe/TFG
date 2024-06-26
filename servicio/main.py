from fastapi import FastAPI, File, UploadFile, BackgroundTasks
import pandas as pd
import io
import logging

# Importar las funciones necesarias
from clasificacion import clasificacion_XGBoost
from regresion import regresion_energiaEntrada
from entrenarModelos import entrenar_modelos

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the classification API. Use /classify or /regress to upload a CSV file."}

@app.post("/classify")
async def TipoParticula(file: UploadFile):
    # Leer el archivo subido
    contents = await file.read()

    # Leer el contenido en un DataFrame de pandas
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

    predictions = clasificacion_XGBoost(df)
        
    # Convertir las predicciones a un formato adecuado para la respuesta
    response = ["It's a kaon" if pred == 1 else "It's a pion" for pred in predictions]
        
    return {"predictions": response}

@app.post("/regress")
async def EnergiaEntrada(file: UploadFile):
    # Leer el archivo subido
    contents = await file.read()

    # Leer el contenido en un DataFrame de pandas
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

    tipo_particula, energia = regresion_energiaEntrada(df)

    if tipo_particula == 0:
        particula = "pion"
    else:
        particula = "kaon"
    
    respuesta = {
        "tipo_particula": particula,
        "energia_entrada": f"{energia} GeV"
    }

    return respuesta

@app.get("/retraining")
async def Retraining(background_tasks: BackgroundTasks):
    background_tasks.add_task(entrenar_modelos)
    return {"Entrenamiento_finalizado"}
