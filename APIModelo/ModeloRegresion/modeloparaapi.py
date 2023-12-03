# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 02:19:13 2023

@author: Daniel
"""

import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score

warnings.simplefilter(action='ignore')

mapeo = {'Diabetes': {'No diabetes': 0, 'Diabetes': 1}, 'Gender': {'male': 0, 'female': 1}}

# Cargar los datos de entrenamiento
data = pd.read_csv('predicciondiabetes.csv')
data.replace(mapeo, inplace=True)

# Eliminar los campos 'id' y 'resultado'
X = data.drop(["Diabetes"], axis=1)
y = data["Diabetes"]

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Entrenar el modelo
logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train, y_train)
y_pred_train = logreg.predict(X_train)

# Mostrar resultados del conjunto de entrenamiento
print('Exactitud del clasificador de regresión logística en el conjunto de entrenamiento: {:.2f}'.format(logreg.score(X_train, y_train)))
print('Precisión del clasificador de regresión logística en el conjunto de entrenamiento: {:.2f}'.format(precision_score(y_train, y_pred_train)))

# Guardar el modelo y el escalador
joblib.dump(logreg, 'modelo_logreg.joblib')

# Entrenar un escalador y guardarlo
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, 'escalador.joblib')

# Hacer predicciones en el conjunto de prueba
y_pred_test = logreg.predict(X_test)

# Mostrar resultados del conjunto de prueba
print('Exactitud del clasificador de regresión logística en el conjunto de prueba: {:.2f}'.format(logreg.score(X_test, y_test)))
print('Precisión del clasificador de regresión logística en el conjunto de prueba: {:.2f}'.format(precision_score(y_test, y_pred_test)))
