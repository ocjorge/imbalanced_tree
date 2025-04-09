# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, learning_curve, GridSearchCV) # Añadido GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
# Necesitarás instalar imbalanced-learn y matplotlib:
# pip install imbalanced-learn matplotlib
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    imblearn_installed = True
except ImportError:
    print("Advertencia: La librería 'imbalanced-learn' no está instalada.")
    print("El modelo MLP con SMOTE no se ejecutará completamente.")
    imblearn_installed = False
    class SMOTE: pass
    class ImbPipeline: pass


# --- Configuración ---
CSV_FILE_PATH = 'cinco.csv'
TARGET_COLUMN = 'Species'
CSV_ENCODING = 'latin-1' # O 'cp1252' si latin-1 falla
TEST_SET_SIZE = 0.3
RANDOM_SEED = 42
N_SPLITS_CV = 2 # Mantenido en 2 por clase SEVERA
LEARNING_CURVE_SCORING = 'f1_weighted'
# Threshold para SelectFromModel: "mean", "median", o un float (e.g., 0.01)
FEATURE_SELECTION_THRESHOLD = "median"
# Profundidad máxima para el Decision Tree para reducir overfitting MÁS AGRESIVAMENTE
DT_MAX_DEPTH = 4
# Métrica para optimizar GridSearchCV (f1_macro prioriza clases minoritarias)
GRIDSEARCH_SCORING = 'f1_macro'

# --- Función para Graficar Curva de Aprendizaje ---
def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=-1,
                        train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'):
    """Genera gráfico de curva de aprendizaje."""
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel(f"Score ({scoring})")
    try:
        train_sizes_abs, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
            scoring=scoring, error_score='raise')
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()
        plt.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes_abs, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes_abs, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes_abs, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.legend(loc="best")
        plt.ylim(bottom=max(-0.05, plt.ylim()[0]))
        plt.tight_layout()
        plt.show()
        print(f"Curva de aprendizaje '{title}' generada.")
    except ValueError as e:
        print(f"\nERROR: No se pudo generar la curva de aprendizaje para '{title}'. Razón: {e}")
        plt.close()
    except Exception as e:
        print(f"\nERROR inesperado al generar la curva de aprendizaje para '{title}': {e}")
        plt.close()

# --- 1. Cargar Datos ---
print(f"--- Cargando datos desde: {CSV_FILE_PATH} (Encoding: {CSV_ENCODING}) ---")
try:
    df = pd.read_csv(CSV_FILE_PATH, encoding=CSV_ENCODING)
    print("CSV cargado exitosamente.")
    print(f"Forma del DataFrame: {df.shape}")
    if TARGET_COLUMN not in df.columns: raise ValueError(f"Columna '{TARGET_COLUMN}' no encontrada.")
    print(f"\nDistribución de '{TARGET_COLUMN}':")
    print(df[TARGET_COLUMN].value_counts())
except Exception as e: print(f"Error Fatal cargando CSV: {e}"); exit()

# --- 2. Separar Características (X) y Etiqueta (y) ---
print(f"\n--- Separando características (X) y etiqueta (y: '{TARGET_COLUMN}') ---")
try:
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    if not np.issubdtype(y.dtype, np.number):
        print(f"Aplicando LabelEncoder a '{TARGET_COLUMN}'.")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        label_mapping = {i: cls for i, cls in enumerate(label_encoder.classes_)}
        print("Mapeo:", label_mapping)
        target_names_report = [str(label_mapping[k]) for k in sorted(label_mapping.keys())]
    else:
        label_encoder = None
        unique_labels = sorted(y.unique())
        label_mapping = {lbl: str(lbl) for lbl in unique_labels}
        target_names_report = [str(lbl) for lbl in unique_labels]
        print(f"'{TARGET_COLUMN}' ya es numérica. Mapeo a string:", label_mapping)
    non_numeric_cols = X.select_dtypes(exclude=np.number).columns
    if len(non_numeric_cols) > 0:
        print(f"Error Fatal: Columnas no numéricas en X: {list(non_numeric_cols)}. Preprocesar.")
        exit()
    print("\nCaracterísticas (X) son numéricas.")
    feature_names = X.columns.tolist()
except Exception as e: print(f"Error Fatal en separación X/y: {e}"); exit()

# --- 3. Dividir en Conjuntos de Entrenamiento y Prueba ---
print("\n--- Dividiendo datos en entrenamiento y prueba ---")
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SET_SIZE, random_state=RANDOM_SEED, stratify=y)
    print(f"Train: {X_train.shape[0]} muestras, Test: {X_test.shape[0]} muestras")
    print("Distribución train (conteo):")
    print(pd.Series(y_train).map(label_mapping).value_counts())
except Exception as e: print(f"Error Fatal en train_test_split: {e}"); exit()

# --- 4. Preprocesamiento: Escalado ---
print("\n--- 4. Escalando características (StandardScaler) ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Escalado completado.")

# --- 5. Selección de Características ---
print(f"\n--- 5. Selección de Características (Threshold: {FEATURE_SELECTION_THRESHOLD}) ---")
try:
    selector_estimator = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, class_weight='balanced', n_jobs=-1)
    selector_estimator.fit(X_train_scaled, y_train)
    selector = SelectFromModel(selector_estimator, threshold=FEATURE_SELECTION_THRESHOLD, prefit=True)
    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    n_features_original = X_train_scaled.shape[1]
    n_features_selected = X_train_selected.shape[1]
    print(f"Características seleccionadas: {n_features_selected} de {n_features_original}")
    selected_mask = selector.get_support()
    selected_features = np.array(feature_names)[selected_mask]
    # print("Características seleccionadas:", selected_features.tolist())
except Exception as e: print(f"Error Fatal en Selección de Características: {e}"); exit()

# --- Definir Validación Cruzada Estratificada ---
cv_strategy = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_SEED)
print(f"\nUsando Validación Cruzada Estratificada ({cv_strategy.get_n_splits()} folds) para evaluación interna...")

# --- 6. Modelo 1: Random Forest (Seleccionado y Optimizado con GridSearchCV) ---
print("\n\n===== Modelo 1: Random Forest Classifier (Features Seleccionadas + Optimización) =====")

# --- 6a. Optimización de Hiperparámetros ---
print(f"\n--- 6a. Optimizando Hiperparámetros para RF (scoring: {GRIDSEARCH_SCORING}) ---")
param_grid_rf = {
    'n_estimators': [50, 100, 150, 200], # Rango un poco más amplio
    'max_depth': [None, 4, 6, 8, 10],   # Explorar profundidades diferentes
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5],
    # 'max_features': ['sqrt', 'log2', None] # Podrías añadir esto si buscas más ajuste
}
rf_for_tuning = RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced', n_jobs=-1)
grid_search_rf = GridSearchCV(estimator=rf_for_tuning, param_grid=param_grid_rf,
                              cv=cv_strategy, scoring=GRIDSEARCH_SCORING, n_jobs=-1, verbose=1)
print("Ejecutando GridSearchCV para RF...")
try:
    grid_search_rf.fit(X_train_selected, y_train) # <<-- Usar datos seleccionados
    print(f"\nMejores parámetros encontrados para RF: {grid_search_rf.best_params_}")
    print(f"Mejor puntuación {GRIDSEARCH_SCORING} (CV): {grid_search_rf.best_score_:.4f}")
    # --- El mejor modelo ya está entrenado en grid_search_rf.best_estimator_ ---
    rf_model_sel = grid_search_rf.best_estimator_
    rf_optimizado_exitoso = True
except Exception as e:
    print(f"\nERROR: GridSearchCV falló para Random Forest: {e}")
    print("Se usará un RF con parámetros por defecto como fallback.")
    # Fallback: usar el modelo no optimizado si GridSearchCV falla
    rf_model_sel = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, class_weight='balanced', n_jobs=-1)
    rf_model_sel.fit(X_train_selected, y_train)
    rf_optimizado_exitoso = False

# --- 6b. Curva de Aprendizaje (del modelo RF optimizado o fallback) ---
if rf_optimizado_exitoso:
    print("\nGenerando Curva de Aprendizaje (RF Optimizado)...")
    plot_learning_curve(rf_model_sel, f"Learning Curve (RF Optimized, Best Score: {grid_search_rf.best_score_:.2f})",
                        X_train_selected, y_train, cv=cv_strategy, n_jobs=-1, scoring=LEARNING_CURVE_SCORING)
else:
     print("\nGenerando Curva de Aprendizaje (RF Fallback)...")
     plot_learning_curve(rf_model_sel, "Learning Curve (RF Fallback/Default)",
                         X_train_selected, y_train, cv=cv_strategy, n_jobs=-1, scoring=LEARNING_CURVE_SCORING)


# --- 6c. Evaluación Final en Test Set (del modelo RF optimizado o fallback) ---
print("\n--- Evaluación Test (RF Seleccionado y Optimizado/Fallback) ---")
y_pred_rf_sel = rf_model_sel.predict(X_test_selected)
print("\nMatriz de Confusión:")
print(pd.DataFrame(confusion_matrix(y_test, y_pred_rf_sel, labels=sorted(label_mapping.keys())), index=target_names_report, columns=target_names_report))
print("\nReporte de Clasificación Detallado:")
print(classification_report(y_test, y_pred_rf_sel, labels=sorted(label_mapping.keys()), target_names=target_names_report, zero_division=0))
print("\nMétricas Generales:")
print(f"  Accuracy:          {accuracy_score(y_test, y_pred_rf_sel):.4f}")
print(f"  F1 Score (Weighted): {f1_score(y_test, y_pred_rf_sel, average='weighted', zero_division=0):.4f}")
print(f"  F1 Score (Macro):    {f1_score(y_test, y_pred_rf_sel, average='macro', zero_division=0):.4f}")


# --- 7. Modelo 2: MLP (SMOTE + Features Seleccionadas) ---
# (Sin cambios respecto a la versión anterior - se mantiene el proceso y las notas)
if imblearn_installed:
    print("\n\n===== Modelo 2: MLP Classifier (SMOTE + Features Seleccionadas) =====")
    print("NOTA: CV y Curva de Aprendizaje omitidas para esta combinación por complejidad/errores previos.")
    try:
        # ... (código SMOTE, scale, select como antes) ...
        print("\nAplicando SMOTE a datos de entrenamiento originales...")
        smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=1)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"Tamaño después de SMOTE: {X_train_resampled.shape[0]} muestras")
        print("Distribución clases post-SMOTE:")
        print(pd.Series(y_train_resampled).map(label_mapping).value_counts())
        print("\nEscalando y Seleccionando Features en datos remuestreados...")
        X_train_resampled_scaled = scaler.transform(X_train_resampled)
        X_train_resampled_selected = selector.transform(X_train_resampled_scaled)
        print(f"Forma final datos entrenamiento MLP: {X_train_resampled_selected.shape}")

        # ... (código MLP como antes) ...
        mlp_model_sel = MLPClassifier(
            hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
            max_iter=500, random_state=RANDOM_SEED, early_stopping=True,
            n_iter_no_change=20, validation_fraction=0.1)
        print("\nEntrenando modelo final MLP (SMOTE + Selección)...")
        mlp_model_sel.fit(X_train_resampled_selected, y_train_resampled)
        print("Entrenamiento completado.")

        # ... (código de evaluación MLP como antes) ...
        print("\n--- Evaluación Test (MLP, SMOTE + Selección) ---")
        y_pred_mlp_sel = mlp_model_sel.predict(X_test_selected)
        print("\nMatriz de Confusión:")
        print(pd.DataFrame(confusion_matrix(y_test, y_pred_mlp_sel, labels=sorted(label_mapping.keys())), index=target_names_report, columns=target_names_report))
        print("\nReporte de Clasificación Detallado:")
        print(classification_report(y_test, y_pred_mlp_sel, labels=sorted(label_mapping.keys()), target_names=target_names_report, zero_division=0))
        print("\nMétricas Generales:")
        print(f"  Accuracy:          {accuracy_score(y_test, y_pred_mlp_sel):.4f}")
        print(f"  F1 Score (Weighted): {f1_score(y_test, y_pred_mlp_sel, average='weighted', zero_division=0):.4f}")
        print(f"  F1 Score (Macro):    {f1_score(y_test, y_pred_mlp_sel, average='macro', zero_division=0):.4f}")

    except Exception as e:
        print(f"\nError durante el proceso MLP con SMOTE y Selección: {e}")
else:
    print("\n\n===== Modelo 2: MLP Classifier (Red Neuronal) - OMITIDO (imblearn no instalado) =====")


# --- 8. Modelo 3: Decision Tree (con Features Seleccionadas y Max Depth) ---
# (Sin cambios respecto a la versión anterior - se mantiene max_depth=4)
print(f"\n\n===== Modelo 3: Decision Tree Classifier (Features Seleccionadas, max_depth={DT_MAX_DEPTH}) =====")
dt_model_instance_sel = DecisionTreeClassifier(random_state=RANDOM_SEED, class_weight='balanced', max_depth=DT_MAX_DEPTH)
print(f"\nGenerando Curva de Aprendizaje (DT Seleccionado, max_depth={DT_MAX_DEPTH})...")
plot_learning_curve(dt_model_instance_sel, f"Learning Curve (DT Sel. Feat., max_depth={DT_MAX_DEPTH})",
                    X_train_selected, y_train, cv=cv_strategy, n_jobs=-1, scoring=LEARNING_CURVE_SCORING)
print("\nEvaluando con CV (DT Seleccionado)...")
try:
    cv_scores_dt_sel = cross_val_score(dt_model_instance_sel, X_train_selected, y_train, cv=cv_strategy, scoring='f1_weighted', n_jobs=-1)
    print(f"  F1 Ponderado Promedio (CV): {cv_scores_dt_sel.mean():.4f} +/- {cv_scores_dt_sel.std():.4f}")
except Exception as e: print(f"  Validación Cruzada falló: {e}")
print("\nEntrenando modelo final DT (Seleccionado)...")
dt_model_sel = dt_model_instance_sel.fit(X_train_selected, y_train)
print("Entrenamiento completado.")
print("\n--- Evaluación Test (DT Seleccionado) ---")
y_pred_dt_sel = dt_model_sel.predict(X_test_selected)
print("\nMatriz de Confusión:")
print(pd.DataFrame(confusion_matrix(y_test, y_pred_dt_sel, labels=sorted(label_mapping.keys())), index=target_names_report, columns=target_names_report))
print("\nReporte de Clasificación Detallado:")
print(classification_report(y_test, y_pred_dt_sel, labels=sorted(label_mapping.keys()), target_names=target_names_report, zero_division=0))
print("\nMétricas Generales:")
print(f"  Accuracy:          {accuracy_score(y_test, y_pred_dt_sel):.4f}")
print(f"  F1 Score (Weighted): {f1_score(y_test, y_pred_dt_sel, average='weighted', zero_division=0):.4f}")
print(f"  F1 Score (Macro):    {f1_score(y_test, y_pred_dt_sel, average='macro', zero_division=0):.4f}")


# --- 9. Ejemplo de Predicción con Nuevos Datos (usando RF Optimizado) ---
print("\n\n===== Ejemplo de Predicción (usando RF Optimizado/Fallback) =====")
# Usar el modelo RF encontrado por GridSearchCV (o el fallback)
modelo_prediccion = rf_model_sel
if rf_optimizado_exitoso:
    nombre_modelo_pred = f"Random Forest (Optimized: {grid_search_rf.best_params_})"
else:
     nombre_modelo_pred = "Random Forest (Fallback/Default, Selected Features)"

print(f"Usando modelo: {nombre_modelo_pred}")
try:
    # 1. Crear datos de ejemplo
    nuevos_datos_dict = {col: [np.random.rand() * 10] for col in feature_names}
    nuevos_datos_df = pd.DataFrame(nuevos_datos_dict)
    # print("\nNuevos datos a predecir (DataFrame original):")
    # print(nuevos_datos_df)

    # 2. Escalar
    nuevos_datos_scaled = scaler.transform(nuevos_datos_df)
    # print("\nNuevos datos escalados:")

    # 3. Seleccionar features
    nuevos_datos_selected = selector.transform(nuevos_datos_scaled)
    print(f"\nNuevos datos con features seleccionadas ({nuevos_datos_selected.shape[1]} features).")

    # 4. Predecir
    prediccion_codificada = modelo_prediccion.predict(nuevos_datos_selected)
    prediccion_proba = modelo_prediccion.predict_proba(nuevos_datos_selected)

    # 5. Decodificar y Mostrar
    prediccion_etiqueta = label_mapping[prediccion_codificada[0]]
    print(f"\nPredicción: {prediccion_etiqueta} (Código: {prediccion_codificada[0]})")
    print("\nProbabilidades estimadas:")
    clases_modelo = modelo_prediccion.classes_
    prob_dict = {label_mapping[clases_modelo[i]]: prob for i, prob in enumerate(prediccion_proba[0])}
    for clase, prob in prob_dict.items(): print(f"  Clase '{clase}': {prob:.4f} ({(prob*100):.2f}%)")

except Exception as e:
    print(f"\nError Fatal en Predicción: {e}")

print("\n\n--- Fin del Script ---")
