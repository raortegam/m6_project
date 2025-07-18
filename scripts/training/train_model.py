"""
Script para entrenar y optimizar un modelo LightGBM con Optuna.

Este script realiza las siguientes tareas:
1. Carga y preprocesa los datos
2. Realiza la optimización de hiperparámetros con Optuna
3. Entrena el modelo final con los mejores parámetros
4. Evalúa el modelo en el conjunto de prueba
5. Guarda el modelo y los resultados
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import json
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

# Añadir el directorio raíz al path para importaciones
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.base_model import LGBMClassifier
from src.models.optimization import LGBMOptimizer
from src.preprocessing.preprocessors import (
    load_config, load_raw_data, preprocess_data,
    track_model_performance, save_processed_data
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(config_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    """
    Carga y preprocesa los datos.
    
    Args:
        config_path: Ruta al archivo de configuración.
        
    Returns:
        Tupla con (X_train, X_test, y_train, y_test, preprocessor, class_weights)
    """
    print("Cargando y preprocesando datos...")
    
    # Cargar configuración
    config = load_config(config_path)
    
    # Cargar datos crudos
    df = load_raw_data()
    
    # Preprocesar datos
    X_train, X_test, y_train, y_test, preprocessor, class_weights = preprocess_data(
        df=df,
        target_column=config.get('data', {}).get('target_column', 'Churn'),
        test_size=config.get('preprocessing', {}).get('test_size', 0.2),
        random_state=config.get('model', {}).get('random_state', 42),
        config_path=config_path
    )
    
    print(f"\nDatos cargados y preprocesados:")
    print(f"- Conjunto de entrenamiento: {X_train.shape[0]} muestras, {X_train.shape[1]} características")
    print(f"- Conjunto de prueba: {X_test.shape[0]} muestras")
    print(f"- Distribución de clases en entrenamiento: {dict(y_train.value_counts())}")
    print(f"- Pesos de clases: {class_weights}")
    
    return X_train, X_test, y_train, y_test, preprocessor, class_weights

def optimize_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series, 
                           class_weights: dict, config: dict) -> Dict[str, Any]:
    """
    Optimiza los hiperparámetros del modelo usando Optuna.
    
    Args:
        X_train: Datos de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        class_weights: Pesos de las clases para manejar el desbalanceo.
        config: Configuración del proyecto.
        
    Returns:
        Diccionario con los mejores parámetros encontrados.
    """
    print("\nIniciando optimización de hiperparámetros con Optuna...")
    
    # Configurar optimizador
    optimizer = LGBMOptimizer()
    
    # Ejecutar optimización
    best_params = optimizer.optimize(
        X_train,
        y_train,
        n_trials=config.get('optimization', {}).get('n_trials', 100),
        timeout=config.get('optimization', {}).get('timeout', 3600)
    )
    
    print("\nOptimización completada. Mejores parámetros encontrados:")
    for param, value in best_params.items():
        print(f"- {param}: {value}")
    
    return best_params

def train_final_model(X_train: pd.DataFrame, y_train: pd.Series,
                     X_val: Optional[pd.DataFrame] = None,
                     y_val: Optional[pd.Series] = None,
                     params: Optional[dict] = None,
                     config: Optional[dict] = None) -> lgb.Booster:
    """
    Entrena el modelo final con los mejores parámetros.
    
    Args:
        X_train: Datos de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        X_val: Datos de validación (opcional).
        y_val: Etiquetas de validación (opcional).
        params: Parámetros del modelo. Si es None, se usan los de la configuración.
        config: Configuración del proyecto.
        
    Returns:
        Modelo entrenado.
    """
    if config is None:
        config = load_config()
    
    if params is None:
        params = config.get('model', {}).get('params', {})
    
    print("\nEntrenando modelo final...")
    
    # Asegurarse de que los datos sean arrays de numpy
    if isinstance(X_train, pd.DataFrame) or isinstance(X_train, pd.Series):
        X_train = X_train.values
    if isinstance(y_train, pd.Series):
        y_train = y_train.values
    if X_val is not None and (isinstance(X_val, pd.DataFrame) or isinstance(X_val, pd.Series)):
        X_val = X_val.values
    if y_val is not None and isinstance(y_val, pd.Series):
        y_val = y_val.values
    
    # Crear datasets
    train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    valid_sets = [train_data]
    valid_names = ['train']
    
    if X_val is not None and y_val is not None:
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)
        valid_sets.append(valid_data)
        valid_names.append('valid')
    
    # Parámetros de entrenamiento
    train_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'random_state': config.get('model', {}).get('random_state', 42),
        **params
    }
    
    # Entrenar modelo
    model = lgb.train(
        params=train_params,
        train_set=train_data,
        valid_sets=valid_sets,
        valid_names=valid_names,
        num_boost_round=config.get('model', {}).get('num_boost_round', 1000),
        early_stopping_rounds=config.get('model', {}).get('early_stopping_rounds', 50),
        verbose_eval=config.get('model', {}).get('verbose_eval', 50)
    )
    
    return model

def evaluate_model(model: lgb.Booster, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evalúa el modelo en el conjunto de prueba.
    
    Args:
        model: Modelo entrenado.
        X_test: Características de prueba.
        y_test: Etiquetas de prueba.
        
    Returns:
        Diccionario con las métricas de evaluación.
    """
    print("\nEvaluando modelo en el conjunto de prueba...")
    
    # Hacer predicciones
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calcular métricas
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'average_precision': average_precision_score(y_test, y_pred_proba)
    }
    
    # Mostrar informe de clasificación
    print("\nInforme de clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Mostrar matriz de confusión con números en negro
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'],
                    annot_kws={'color': 'black', 'weight': 'bold'})
    
    # Asegurar que el texto sea negro
    for text in ax.texts:
        text.set_color('black')
        
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    plt.show()
    
    return metrics

def save_model_and_artifacts(model: lgb.Booster, metrics: dict, params: dict,
                           config: dict, preprocessor: Any = None) -> None:
    """
    Guarda el modelo y los artefactos asociados.
    
    Args:
        model: Modelo entrenado.
        metrics: Métricas de evaluación.
        params: Parámetros del modelo.
        config: Configuración del proyecto.
        preprocessor: Objeto preprocesador (opcional).
    """
    # Crear directorios necesarios
    model_dir = Path(config.get('model', {}).get('model_dir', 'models'))
    model_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Guardar modelo
    model_path = model_dir / f'model_{timestamp}.pkl'
    joblib.dump(model, model_path)
    
    # Guardar preprocesador
    if preprocessor is not None:
        preprocessor_path = model_dir / f'preprocessor_{timestamp}.pkl'
        joblib.dump(preprocessor, preprocessor_path)
    
    # Guardar métricas
    metrics_path = model_dir / f'metrics_{timestamp}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Guardar parámetros
    params_path = model_dir / f'params_{timestamp}.json'
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)
    
    # Registrar en el archivo de seguimiento
    track_model_performance(
        model_name=f'LightGBM_{timestamp}',
        metrics=metrics,
        params=params,
        filepath=model_dir / 'model_tracking.csv'
    )
    
    print("\nModelo y artefactos guardados en:")
    print(f"- Modelo: {model_path}")
    if preprocessor is not None:
        print(f"- Preprocesador: {preprocessor_path}")
    print(f"- Métricas: {metrics_path}")
    print(f"- Parámetros: {params_path}")

def parse_args():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Entrenar y optimizar modelo LightGBM')
    parser.add_argument('--optimize', action='store_true',
                      help='Realizar optimización de hiperparámetros con Optuna')
    parser.add_argument('--config', type=str, default=None,
                      help='Ruta al archivo de configuración')
    parser.add_argument('--n-trials', type=int, default=None,
                      help='Número de pruebas para la optimización')
    parser.add_argument('--timeout', type=int, default=None,
                      help='Tiempo máximo de optimización en segundos')
    
    return parser.parse_args()


def main():
    # Parsear argumentos
    args = parse_args()
    
    try:
        # Cargar configuración
        config = load_config(args.config)
        
        # Sobrescribir configuración con argumentos de línea de comandos
        if args.n_trials is not None:
            config['optimization']['n_trials'] = args.n_trials
        if args.timeout is not None:
            config['optimization']['timeout'] = args.timeout
        
        # Cargar y preprocesar datos
        X_train, X_test, y_train, y_test, preprocessor, class_weights = load_and_preprocess_data(args.config)
        
        # Dividir datos de entrenamiento en entrenamiento y validación
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=config.get('model', {}).get('random_state', 42),
            stratify=y_train
        )
        
        # Optimizar hiperparámetros si se solicita
        if args.optimize:
            best_params = optimize_hyperparameters(
                X_train_final, y_train_final, class_weights, config
            )
        else:
            print("\nUsando parámetros por defecto.")
            best_params = config.get('model', {}).get('params', {})
        
        # Añadir pesos de clases si no están en los parámetros
        if 'scale_pos_weight' not in best_params and class_weights is not None:
            best_params['scale_pos_weight'] = class_weights[1] / class_weights[0]  # ratio de clases
        
        # Entrenar modelo final con todos los datos de entrenamiento
        model = train_final_model(
            X_train=X_train,  # Usar todos los datos de entrenamiento
            y_train=y_train,
            X_val=X_val,      # Usar conjunto de validación para early stopping
            y_val=y_val,
            params=best_params,
            config=config
        )
        
        # Evaluar modelo en el conjunto de prueba
        metrics = evaluate_model(model, X_test, y_test)
        
        # Guardar modelo y artefactos
        save_model_and_artifacts(
            model=model,
            metrics=metrics,
            params=best_params,
            config=config,
            preprocessor=preprocessor
        )
        
        print("\n¡Entrenamiento completado exitosamente!")
        
    except Exception as e:
        print(f"\nError durante el entrenamiento: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
