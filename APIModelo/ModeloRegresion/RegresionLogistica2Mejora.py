# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 22:04:40 2023

@author: Daniel
"""

import joblib
import warnings
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import numpy as np
from string import ascii_letters
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
warnings.simplefilter(action='ignore') 

mapeo = {'Diabetes': {'No diabetes': 0, 'Diabetes': 1}, 'Gender':{'male': 0, 'female': 1}}


data = pd.read_csv('predicciondiabetes.csv')
data.head()

data.replace(mapeo, inplace=True)

data.shape

missing_values = data.isnull().sum()
print("Missing Values:")
print(missing_values)


data["Diabetes"].value_counts()

#percentage distribution of the "Diabetes" 
print(100 * data["Diabetes"].value_counts() / len(data))

with_diabetes = data['Diabetes'].value_counts()[1]
without_diabetes = data['Diabetes'].value_counts()[0]
print(f"Pacientes con Diabetes: {with_diabetes}\nPacientes sin Diabetes: {without_diabetes}")

sns.countplot(x="Diabetes", data=data)
plt.show()


#visualizing Pregnancies avrage Diabetes
plt.figure(figsize=(8,6))
sns.countplot(x='Gender', hue='Diabetes', data = data)
plt.show()


data.describe().T


plt.figure(figsize=(16, 8))
for i, col in enumerate(data.columns[:-1]):
    plt.subplot(4, 4, i + 1)
    sns.histplot(data[data['Diabetes'] == 1][col], kde=True, label='Diabetes', color='blue')
    sns.histplot(data[data['Diabetes'] == 0][col], kde=True, label='No Diabetes', color='orange')
    plt.title(f"Distribution of {col}")
    plt.legend()
plt.tight_layout()
plt.show()


data.groupby('Diabetes').mean()



plt.figure(figsize=(16,8))

for i, col in enumerate(data.columns[:-1]):
    plt.subplot(4, 4, i + 1)
    sns.barplot(x='Diabetes', y=data[col], data=data)
    plt.title(f"{col} vs. Diabetes")

plt.tight_layout()
plt.show()



data.corr()




plt.figure(figsize=(10, 6))

sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Quitar los campos 'id' y 'resultado'
X = data.drop(["Diabetes"], axis=1)
y = data["Diabetes"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state= 0)




logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)



print("Exactitud: ", metrics.accuracy_score(y_test, y_pred))


"""
fqr, tpr, _ = metrics.roc_curve(y_test, y_pred)
plt.plot(fqr,tpr,label="data 1")
plt.legend(loc=4)
plt.show()



y_pred_proba = logreg.predict_proba(X_test)[::,1]
fqr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
plt.plot(fqr,tpr, label='data 1')
plt.legend(loc=4)
plt.show()

"""

fqr, tpr, _ = metrics.roc_curve(y_test, y_pred)
plt.plot(fqr,tpr,label="data 1")
plt.xlabel("Tasa de falsos positivos (FPR)")
plt.ylabel("Tasa de verdaderos positivos (TPR)")
plt.title("Curva ROC")
plt.legend(loc=4)
plt.show()

y_pred_proba = logreg.predict_proba(X_test)[::,1]
fqr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
plt.plot(fqr,tpr, label='data 1')
plt.xlabel("Tasa de falsos positivos (FPR)")
plt.ylabel("Tasa de verdaderos positivos (TPR)")
plt.title("Curva ROC")
plt.show()


df3 = pd.read_csv("predicciondiabetes.csv")

df3 = pd.get_dummies(df3, columns=['Gender'])

mapeo = {'Diabetes': {'No diabetes': 0, 'Diabetes': 1}}
df3.replace(mapeo, inplace=True)

X1 = df3.drop('Diabetes', axis=1)
y1 = df3[['Diabetes']]
y1 = y1.values.ravel()

# Train Test Split (test_size=0.25)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.25, random_state=0)

# Create a MinMaxScaler instance and fit it to the training data
scaler = MinMaxScaler() # saga solver requires features to be scaled for model conversion
X1_train = scaler.fit_transform(X1_train)

# Transform the test data using the same scaler
X1_test = scaler.transform(X1_test)

# Create a LogisticRegression instance and fit it to the scaled training data
logreg1 = LogisticRegression() # (default penalty = "l2") / (default solver = "lbfgs")
logreg1.fit(X1_train, y1_train)

# Predict using the scaled test data
y1_pred = logreg1.predict(X1_test)

print('Exactitud del clasificador de regresión logística en el conjunto de entrenamiento: {:.2f}'.format(logreg1.score(X1_train, y1_train)))
print('Exactitud del clasificador de regresión logística en el conjunto de prueba: {:.2f}'.format(logreg1.score(X1_test, y1_test)))
print('Precisión del clasificador de regresión logística en el conjunto de prueba: {:.2f}'.format(precision_score(y1_test, y1_pred)))


# Guardar el modelo y el escalador
joblib.dump(logreg, 'modelo_logreg.joblib')
joblib.dump(scaler, 'escalador.joblib')