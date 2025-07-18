"""
Módulo para la definición del modelo base de LightGBM.
"""

from typing import Dict, Any, Optional
import lightgbm as lgb
import numpy as np
from pathlib import Path
import joblib
import yaml

class LGBMClassifier:
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa el clasificador LightGBM.
        
        Args:
            config_path: Ruta al archivo de configuración.
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.feature_importances_ = None
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Carga la configuración desde el archivo YAML."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / 'config.yaml'
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Entrena el modelo LightGBM.
        
        Args:
            X_train: Datos de entrenamiento.
            y_train: Etiquetas de entrenamiento.
            X_val: Datos de validación (opcional).
            y_val: Etiquetas de validación (opcional).
            
        Returns:
            El modelo entrenado.
        """
        # Configurar los parámetros del modelo
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'random_state': self.config.get('model', {}).get('random_state', 42),
            'verbose': -1,
            **self.config.get('model', {}).get('params', {})
        }
        
        # Crear el dataset de LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # Si hay datos de validación, usarlos para early stopping
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(valid_data)
            valid_names.append('valid')
        
        # Entrenar el modelo
        self.model = lgb.train(
            params=params,
            train_set=train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            num_boost_round=self.config.get('model', {}).get('num_boost_round', 1000),
            early_stopping_rounds=self.config.get('model', {}).get('early_stopping_rounds', 50),
            verbose_eval=self.config.get('model', {}).get('verbose_eval', 50)
        )
        
        # Guardar importancia de características
        self.feature_importances_ = self.model.feature_importance(importance_type='gain')
        
        return self.model
    
    def predict_proba(self, X):
        """Realiza predicciones de probabilidad."""
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado. Llame al método 'train' primero.")
        return self.model.predict(X)
    
    def save_model(self, filepath: str):
        """Guarda el modelo en un archivo."""
        if self.model is None:
            raise ValueError("No hay modelo para guardar.")
        joblib.dump(self.model, filepath)
    
    @classmethod
    def load_model(cls, filepath: str):
        """Carga un modelo desde un archivo."""
        model = cls()
        model.model = joblib.load(filepath)
        return model
