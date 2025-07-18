"""
Módulo de preprocesamiento de datos.

Este módulo contiene funciones para el preprocesamiento de datos, incluyendo
la identificación de variables categóricas, codificación one-hot, manejo de
valores nulos y división de conjuntos de entrenamiento y prueba.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Union, Optional, Any
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import joblib


def load_config(config_path: Union[str, Path] = None) -> Dict:
    """
    Carga la configuración desde un archivo YAML.

    Args:
        config_path: Ruta al archivo de configuración. Si es None, intenta cargar
                   desde 'config.yaml' en el directorio raíz.

    Returns:
        dict: Configuración cargada.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'config.yaml'
    else:
        config_path = Path(config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_raw_data(filepath: Union[str, Path] = None) -> pd.DataFrame:
    """
    Carga los datos crudos del archivo CSV.
    
    Args:
        filepath: Ruta al archivo CSV. Si es None, usa la ruta por defecto.
        
    Returns:
        DataFrame con los datos cargados.
    """
    if filepath is None:
        filepath = Path(__file__).parent.parent.parent / 'src' / 'database' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    # Cargar datos
    df = pd.read_csv(filepath)
    
    # Convertir TotalCharges a numérico, manejando errores
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Eliminar filas con valores nulos (si las hay)
    df = df.dropna()
    
    return df


def get_feature_types(df: pd.DataFrame, 
                     target_column: str = None,
                     categorical_threshold: int = 10) -> Tuple[list, list]:
    """
    Identifica automáticamente columnas numéricas y categóricas en un DataFrame.

    Args:
        df: DataFrame de entrada.
        target_column: Nombre de la columna objetivo (a excluir).
        categorical_threshold: Número máximo de valores únicos para considerar 
                             una columna como categórica.

    Returns:
        tuple: (numeric_cols, categorical_cols)
    """
    # Excluir la columna objetivo si se especifica
    exclude = [target_column] if target_column and target_column in df.columns else []
    
    numeric_cols = []
    categorical_cols = []
    
    for col in df.columns:
        if col in exclude:
            continue
            
        # Considerar como categórica si es de tipo objeto o tiene pocos valores únicos
        if df[col].dtype == 'object' or df[col].nunique() <= categorical_threshold:
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)
    
    return numeric_cols, categorical_cols


def create_preprocessing_pipeline(numeric_cols: list, 
                                categorical_cols: list,
                                config: dict) -> ColumnTransformer:
    """
    Crea un pipeline de preprocesamiento para características numéricas y categóricas.
    
    Args:
        numeric_cols: Lista de columnas numéricas.
        categorical_cols: Lista de columnas categóricas.
        config: Diccionario de configuración.
        
    Returns:
        ColumnTransformer configurado.
    """
    # Pipeline para características numéricas
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline para características categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', 
                               drop=config.get('preprocessing', {}).get('one_hot', {}).get('drop', 'first')))
    ])
    
    # Combinar transformadores
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor


def preprocess_data(df: pd.DataFrame, 
                   target_column: str = None,
                   test_size: float = 0.2,
                   random_state: int = 42,
                   config_path: Union[str, Path] = None) -> Tuple[pd.DataFrame, 
                                                               pd.Series, 
                                                               pd.DataFrame, 
                                                               pd.Series,
                                                               dict]:
    """
    Preprocesa los datos aplicando transformaciones a las variables
    y dividiendo en conjuntos de entrenamiento y prueba.

    Args:
        df: DataFrame con los datos a preprocesar.
        target_column: Nombre de la columna objetivo.
        test_size: Proporción del conjunto de prueba.
        random_state: Semilla para la reproducibilidad.
        config_path: Ruta al archivo de configuración.

    Returns:
        Tuple: (X_train, X_test, y_train, y_test, preprocessor, class_weights)
    """
    # Cargar configuración
    config = load_config(config_path)
    
    # Obtener parámetros de configuración
    if target_column is None:
        target_column = config.get('data', {}).get('target_column', 'Churn')
    
    # Convertir target a numérico primero
    if df[target_column].dtype == 'object':
        df[target_column] = df[target_column].map({'Yes': 1, 'No': 0}).astype(int)
    
    # Identificar tipos de columnas (excluyendo el target)
    numeric_cols, categorical_cols = get_feature_types(
        df.drop(columns=[target_column]), 
        categorical_threshold=config.get('preprocessing', {}).get('categorical_threshold', 10)
    )
    
    # Separar características y objetivo
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y  # Estratificar para mantener la proporción de clases
    )
    
    # Crear pipeline de preprocesamiento
    preprocessor = create_preprocessing_pipeline(numeric_cols, categorical_cols, config)
    
    # Ajustar transformador a los datos de entrenamiento
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Asegurarse de que los datos sean numéricos
    if not isinstance(X_train_processed, np.ndarray):
        X_train_processed = X_train_processed.toarray()
    if not isinstance(X_test_processed, np.ndarray):
        X_test_processed = X_test_processed.toarray()
        
    # Convertir a float32 para compatibilidad con LightGBM
    X_train_processed = X_train_processed.astype(np.float32)
    X_test_processed = X_test_processed.astype(np.float32)
    
    # Convertir a DataFrames para mejor manejo
    # Obtener nombres de características después de one-hot encoding
    try:
        # Para versiones recientes de scikit-learn
        cat_processor = preprocessor.named_transformers_['cat']
        if hasattr(cat_processor.named_steps['onehot'], 'get_feature_names_out'):
            cat_features = cat_processor.named_steps['onehot'].get_feature_names_out(categorical_cols)
        else:
            # Para versiones más antiguas
            num_cat_features = X_train_processed.shape[1] - len(numeric_cols)
            cat_features = [f"cat_{i}" for i in range(num_cat_features)]
        
        # Asegurarse de que las características categóricas sean una lista
        cat_features = list(cat_features)
        
        # Verificar que las dimensiones coincidan
        expected_columns = len(numeric_cols) + len(cat_features)
        if X_train_processed.shape[1] != expected_columns:
            # Ajustar el número de columnas si es necesario
            if X_train_processed.shape[1] < expected_columns:
                # Rellenar con ceros si faltan columnas
                padding = np.zeros((X_train_processed.shape[0], expected_columns - X_train_processed.shape[1]))
                X_train_processed = np.hstack([X_train_processed, padding])
                X_test_processed = np.hstack([X_test_processed, np.zeros((X_test_processed.shape[0], expected_columns - X_test_processed.shape[1]))])
            else:
                # Recortar si hay columnas de más
                X_train_processed = X_train_processed[:, :expected_columns]
                X_test_processed = X_test_processed[:, :expected_columns]
        
        feature_names = numeric_cols + cat_features
        
        X_train_processed = pd.DataFrame(
            X_train_processed,
            columns=feature_names,
            index=X_train.index
        )
        
        X_test_processed = pd.DataFrame(
            X_test_processed,
            columns=feature_names,
            index=X_test.index
        )
        
    except Exception as e:
        print(f"Error al obtener nombres de características: {str(e)}")
        # Si hay algún error, crear DataFrames sin nombres de columnas
        X_train_processed = pd.DataFrame(X_train_processed, index=X_train.index)
        X_test_processed = pd.DataFrame(X_test_processed, index=X_test.index)
    
    # Aplicar muestreo para manejar el desbalanceo de clases
    sampling_config = config.get('preprocessing', {}).get('class_imbalance', {})
    sampling_strategy = sampling_config.get('strategy')
    
    if sampling_strategy in ['oversampling', 'undersampling']:
        print(f"\nAplicando {sampling_strategy} para manejar el desbalanceo de clases...")
        
        # Configurar el muestreador según la estrategia
        if sampling_strategy == 'oversampling':
            sampler = RandomOverSampler(
                sampling_strategy=sampling_config.get('oversampling', {}).get('sampling_strategy', 'auto'),
                random_state=sampling_config.get('oversampling', {}).get('random_state', random_state)
            )
        else:  # undersampling
            sampler = RandomUnderSampler(
                sampling_strategy=sampling_config.get('undersampling', {}).get('sampling_strategy', 'auto'),
                random_state=sampling_config.get('undersampling', {}).get('random_state', random_state)
            )
        
        # Aplicar el muestreo
        X_train_processed, y_train = sampler.fit_resample(X_train_processed, y_train)
        
        # Convertir de nuevo a DataFrame si es necesario
        if isinstance(X_train_processed, np.ndarray):
            X_train_processed = pd.DataFrame(
                X_train_processed,
                columns=feature_names if 'feature_names' in locals() else None,
                index=range(len(X_train_processed))
            )
        
        print(f"Tamaño del conjunto de entrenamiento después de {sampling_strategy}:")
        print(f"- Clase 0: {(y_train == 0).sum()} muestras")
        print(f"- Clase 1: {(y_train == 1).sum()} muestras")
    
    # Calcular pesos de clases para manejo de desbalanceo
    class_weights = calculate_class_weights(y_train)
    
    # Guardar el preprocesador
    preprocessor_path = Path(config.get('data', {}).get('processed_path', 'data/processed')) / 'preprocessor.pkl'
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, preprocessor_path)
    
    # Guardar los datos procesados
    save_processed_data(X_train_processed, X_test_processed, y_train, y_test)
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, class_weights


def calculate_class_weights(y: pd.Series) -> dict:
    """
    Calcula los pesos de las clases para manejar el desbalanceo.
    
    Args:
        y: Serie con las etiquetas de clase.
        
    Returns:
        dict: Diccionario con los pesos de cada clase.
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))


def save_processed_data(X_train: pd.DataFrame, 
                      X_test: pd.DataFrame, 
                      y_train: pd.Series, 
                      y_test: pd.Series,
                      output_dir: Union[str, Path] = None) -> None:
    """
    Guarda los datos preprocesados en archivos CSV.

    Args:
        X_train: Conjunto de características de entrenamiento.
        X_test: Conjunto de características de prueba.
        y_train: Variable objetivo de entrenamiento.
        y_test: Variable objetivo de prueba.
        output_dir: Directorio de salida. Si es None, se usará 'data/processed'.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    else:
        output_dir = Path(output_dir)
    
    # Crear directorio si no existe
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar datos
    X_train.to_csv(output_dir / 'X_train.csv', index=False)
    if X_test is not None:
        X_test.to_csv(output_dir / 'X_test.csv', index=False)
    if y_train is not None:
        y_train.to_csv(output_dir / 'y_train.csv', index=False)
    if y_test is not None:
        y_test.to_csv(output_dir / 'y_test.csv', index=False)
    
    print(f"Datos guardados en: {output_dir}")


def track_model_performance(model_name: str, 
                          metrics: dict, 
                          params: dict,
                          filepath: Union[str, Path] = None) -> None:
    """
    Guarda el rendimiento del modelo en un archivo CSV para seguimiento.
    
    Args:
        model_name: Nombre del modelo.
        metrics: Diccionario con las métricas de rendimiento.
        params: Diccionario con los parámetros del modelo.
        filepath: Ruta al archivo de seguimiento. Si es None, se usará 'models/model_tracking.csv'.
    """
    import pandas as pd
    from datetime import datetime
    
    if filepath is None:
        filepath = Path(__file__).parent.parent.parent / 'models' / 'model_tracking.csv'
    else:
        filepath = Path(filepath)
    
    # Crear directorio si no existe
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Crear DataFrame con los resultados
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': model_name,
        **metrics,
        **params
    }
    
    # Convertir a DataFrame
    results_df = pd.DataFrame([results])
    
    # Si el archivo existe, cargarlo y concatenar
    if filepath.exists():
        existing_df = pd.read_csv(filepath)
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
    
    # Guardar resultados
    results_df.to_csv(filepath, index=False)
    print(f"Resultados guardados en: {filepath}")


if __name__ == "__main__":
    # Cargar configuración
    config = load_config()
    
    # Cargar datos crudos
    print("Cargando datos crudos...")
    df = load_raw_data()
    
    # Preprocesar datos
    print("Preprocesando datos...")
    X_train, X_test, y_train, y_test, preprocessor, class_weights = preprocess_data(
        df=df,
        target_column=config.get('data', {}).get('target_column', 'Churn'),
        test_size=config.get('preprocessing', {}).get('test_size', 0.2),
        random_state=config.get('model', {}).get('random_state', 42)
    )
    
    print("\nPreprocesamiento completado exitosamente.")
    print(f"Tamaño del conjunto de entrenamiento: {len(X_train)}")
    print(f"Tamaño del conjunto de prueba: {len(X_test)}")
    print(f"Pesos de clases: {class_weights}")
