# Análisis y Clasificación de Datos con Modelos de Machine Learning

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/travis/com/usuario/proyecto?style=flat)](https://travis-ci.org/usuario/proyecto)
[![Code Quality](https://img.shields.io/codeclimate/maintainability/usuario/proyecto)](https://codeclimate.com/github/usuario/proyecto/maintainability)

Este proyecto realiza la clasificación de un conjunto de datos utilizando diferentes modelos de **Machine Learning**, aplicando técnicas de selección de características, validación cruzada, optimización de hiperparámetros, y evaluación de modelos. El objetivo es entrenar modelos y obtener sus métricas de desempeño, tales como `accuracy`, `F1 score`, y otros indicadores clave. También se explora el uso de técnicas avanzadas como SMOTE para balancear las clases en el conjunto de datos.

## Características

- **Preprocesamiento de Datos:**
  - Carga de datos desde archivos CSV.
  - División de características (`X`) y etiquetas (`y`).
  - Escalado de características con `StandardScaler`.

- **Selección de Características:**
  - Selección utilizando un clasificador `RandomForest`.
  - Umbral de selección configurable.

- **Modelos Implementados:**
  - **Random Forest**: Optimización de hiperparámetros usando `GridSearchCV`.
  - **MLP Classifier**: Implementación con remuestreo SMOTE (si la librería `imbalanced-learn` está instalada).
  - **Decision Tree**: Árbol de decisión con control de profundidad máxima.

- **Evaluación:**
  - Métricas como `F1 score`, `accuracy`, y matrices de confusión.
  - Curvas de aprendizaje para visualización del rendimiento.

## Requisitos

- Python 3.x
- Librerías necesarias:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `imbalanced-learn` (opcional, para SMOTE)
  - `matplotlib`

Instala las dependencias usando:

```bash
pip install -r requirements.txt
```

## Uso

### 1. Cargar y Preprocesar los Datos

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('cinco.csv')
X = df.drop('Species', axis=1)
y = df['Species']

# Escalar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 2. Selección de Características

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Selección de características
model = RandomForestClassifier()
model.fit(X_scaled, y)

selector = SelectFromModel(model)
X_selected = selector.transform(X_scaled)
```

### 3. Entrenamiento y Evaluación del Modelo

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

# Entrenar modelo
model.fit(X_train, y_train)

# Predicciones y evaluación
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

## Contribuciones

Las contribuciones son bienvenidas. Si deseas contribuir, sigue estos pasos:

1. Haz un fork de este repositorio.
2. Crea una rama con tu característica (`git checkout -b nueva-caracteristica`).
3. Realiza tus cambios y haz commit (`git commit -am 'Añadir nueva característica'`).
4. Empuja a tu rama (`git push origin nueva-caracteristica`).
5. Crea un pull request.

## Licencia

Este proyecto está bajo la Licencia MIT.

