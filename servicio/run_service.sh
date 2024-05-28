#!/bin/bash

python3 clasificacion.py
#python3 regresion.py
uvicorn main:app --reload

