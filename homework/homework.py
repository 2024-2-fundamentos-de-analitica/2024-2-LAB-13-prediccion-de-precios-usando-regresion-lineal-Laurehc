#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

import os
import json
import gzip
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Definir rutas
ruta_datos = "files/input"
ruta_modelo = "files/models/model.pkl.gz"
ruta_metricas = "files/output/metrics.json"

# Crear directorios si no existen
os.makedirs(os.path.dirname(ruta_modelo), exist_ok=True)
os.makedirs(os.path.dirname(ruta_metricas), exist_ok=True)

# Función para cargar datos desde archivos ZIP
def cargar_datos(ruta_archivo):
    return pd.read_csv(ruta_archivo, compression="zip")

# Cargar datos
train_df = cargar_datos(os.path.join(ruta_datos, "train_data.csv.zip"))
test_df = cargar_datos(os.path.join(ruta_datos, "test_data.csv.zip"))

# Paso 1: Preprocesamiento de datos
train_df["Age"] = 2021 - train_df["Year"]
test_df["Age"] = 2021 - test_df["Year"]

train_df.drop(columns=["Year", "Car_Name"], inplace=True)
test_df.drop(columns=["Year", "Car_Name"], inplace=True)

# Paso 2: Dividir en X e y
X_train, y_train = train_df.drop(columns=["Selling_Price"]), train_df["Selling_Price"]
X_test, y_test = test_df.drop(columns=["Selling_Price"]), test_df["Selling_Price"]


# Identificar características categóricas y numéricas
categorical_features = ["Fuel_Type", "Selling_type", "Transmission"]
numerical_features = [col for col in X_train.columns if col not in categorical_features]

# Paso 3: Definir el Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("selector", SelectKBest(score_func=f_regression, k=8)),  # Selecciona las 8 mejores variables
        ("regressor", LinearRegression()),
    ]
)

# Paso 4: Optimización de hiperparámetros con validación cruzada
param_grid = {
    "selector__k": [5, 8, 10]  # Probar selección de 5, 8 y 10 mejores variables
}

grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring="neg_mean_absolute_error", n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Paso 5: Guardar el modelo
with gzip.open(ruta_modelo, "wb") as f:
    joblib.dump(best_model, f)

# Paso 6: Evaluación del modelo
def calcular_metricas(y_real, y_pred, dataset):
    return {
        "type": "metrics",
        "dataset": dataset,
        "r2": r2_score(y_real, y_pred),
        "mse": mean_squared_error(y_real, y_pred),
        "mad": mean_absolute_error(y_real, y_pred),
    }

# Calcular métricas
metrics = [
    calcular_metricas(y_train, best_model.predict(X_train), "train"),
    calcular_metricas(y_test, best_model.predict(X_test), "test"),
]

# Guardar métricas
with open(ruta_metricas, "w") as f:
    json.dump(metrics, f, indent=4)



