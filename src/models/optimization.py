"""
Módulo para la optimización de hiperparámetros con Optuna.
"""

import os
import sys
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, average_precision_score
)
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, List, Union, Optional
from datetime import datetime

class LGBMOptimizer:
    def __init__(self, config_path: str = None):
        """
        Inicializa el optimizador de hiperparámetros para LightGBM.
        
        Args:
            config_path: Ruta al archivo de configuración.
        """
        self.config = self._load_config(config_path)
        self.study = None
        self.best_params = None
        self.best_score = None
        self.trials_df = None
        self._setup_directories()
    
    def _setup_directories(self):
        """Configura los directorios necesarios."""
        # Directorio para guardar estudios de Optuna
        self.optuna_dir = Path(self.config.get('model', {}).get('optuna_dir', 'models/optuna'))
        self.optuna_dir.mkdir(parents=True, exist_ok=True)
        
        # Directorio para guardar los modelos temporales
        self.temp_models_dir = Path(self.config.get('model', {}).get('temp_models_dir', 'models/temp'))
        self.temp_models_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Carga la configuración desde el archivo YAML."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / 'config.yaml'
        else:
            config_path = Path(config_path)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _get_parameter_suggestions(self, trial, param_name: str, param_range: list) -> Any:
        """Obtiene sugerencias de parámetros según su tipo y rango."""
        if len(param_range) == 2:
            if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                if param_name in ['num_leaves', 'max_depth']:
                    return trial.suggest_int(param_name, param_range[0], param_range[1], log=True)
                else:
                    return trial.suggest_int(param_name, param_range[0], param_range[1])
            else:
                return trial.suggest_float(param_name, param_range[0], param_range[1], log=True)
        else:
            return trial.suggest_categorical(param_name, param_range)
    
    def _create_model_params(self, trial, X, y) -> dict:
        """Crea el diccionario de parámetros para el modelo."""
        # Parámetros base
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'random_state': self.config.get('model', {}).get('random_state', 42),
            'verbose': -1,
            'is_unbalance': False,  # Usamos scale_pos_weight en su lugar
        }
        
        # Obtener rangos de parámetros de la configuración
        params_range = self.config.get('optimization', {}).get('params_range', {})
        
        # Añadir parámetros sugeridos por Optuna
        for param_name, param_range in params_range.items():
            # Para scale_pos_weight, usar el rango definido en la configuración
            params[param_name] = self._get_parameter_suggestions(trial, param_name, param_range)
            
            # Imprimir el valor de scale_pos_weight si es que se está usando
            if param_name == 'scale_pos_weight':
                print(f"  - {param_name}: {params[param_name]:.2f} (rango: {param_range[0]} a {param_range[1]})")
        
        return params
    
    def _evaluate_model(self, params: dict, X_train, y_train, X_val, y_val) -> dict:
        """Evalúa un modelo con los parámetros dados."""
        # Crear datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Entrenar modelo
        model = lgb.train(
            params=params,
            train_set=train_data,
            valid_sets=[valid_data],
            valid_names=['valid'],
            num_boost_round=self.config.get('model', {}).get('num_boost_round', 1000),
            early_stopping_rounds=self.config.get('model', {}).get('early_stopping_rounds', 50),
            verbose_eval=False,
            class_weight='balanced'
        )
        
        # Hacer predicciones
        y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calcular métricas
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1': f1_score(y_val, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_val, y_pred_proba),
            'average_precision': average_precision_score(y_val, y_pred_proba),
            'log_loss': model.best_score['valid']['binary_logloss']
        }
        
        return metrics, model
    
    def objective(self, trial, X, y) -> float:
        """
        Función objetivo para la optimización con Optuna.
        
        Args:
            trial: Objeto de prueba de Optuna.
            X: Características de entrenamiento.
            y: Etiquetas de entrenamiento.
            
        Returns:
            Valor de la métrica objetivo.
        """
        # Obtener parámetros del modelo
        params = self._create_model_params(trial, X, y)
        
        # Configurar validación cruzada
        cv = StratifiedKFold(
            n_splits=self.config.get('optimization', {}).get('n_splits', 5),
            shuffle=True,
            random_state=self.config.get('model', {}).get('random_state', 42)
        )
        
        # Listas para almacenar métricas
        cv_scores = []
        all_metrics = []
        
        # Validación cruzada
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Evaluar modelo
            metrics, model = self._evaluate_model(params, X_train, y_train, X_val, y_val)
            
            # Guardar métricas
            cv_scores.append(metrics['log_loss'])  # Usamos log_loss como métrica principal para Optuna
            all_metrics.append(metrics)
            
            # Guardar el modelo temporalmente si es el mejor hasta ahora
            if not hasattr(self, 'best_score') or metrics['roc_auc'] > getattr(self, 'best_score', 0):
                self.best_score = metrics['roc_auc']
                model_path = self.temp_models_dir / 'best_model_temp.pkl'
                joblib.dump(model, model_path)
        
        # Calcular métricas promedio
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        
        # Registrar métricas en el trial para análisis posterior
        for metric_name, metric_value in avg_metrics.items():
            trial.set_user_attr(metric_name, float(metric_value))
        
        # Devolver la métrica principal (log_loss para optimización)
        return np.mean(cv_scores)
    
    def optimize(self, X, y, n_trials: int = 100, timeout: int = None, n_splits: int = 5) -> dict:
        """
        Ejecuta la optimización de hiperparámetros.
        
        Args:
            X: Características de entrenamiento.
            y: Etiquetas de entrenamiento.
            n_trials: Número de pruebas para la optimización.
            timeout: Tiempo máximo de optimización en segundos.
            n_splits: Número de divisiones para la validación cruzada.
            
        Returns:
            Diccionario con los mejores parámetros encontrados.
        """
        print("\n" + "="*50)
        print(f"INICIANDO OPTIMIZACIÓN CON {n_trials} TRIALS")
        print("="*50)
        
        try:
            # Configurar el estudio de Optuna con logging estándar
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=self.config.get('model', {}).get('random_state', 42)),
                pruner=HyperbandPruner()
            )
            
            # Habilitar el logging de Optuna
            optuna.logging.set_verbosity(optuna.logging.INFO)
            
            # Callback para mostrar el progreso
            def log_trial(study, trial):
                if study.best_trial.number == trial.number:
                    print(f"\n{'='*50}")
                    print(f"🚀 Nuevo mejor trial encontrado (Trial {trial.number}):")
                    print(f"   Valor: {trial.value:.4f}")
                    print("   Parámetros:")
                    for key, value in trial.params.items():
                        print(f"   - {key}: {value}")
                    print(f"{'='*50}")
        
            # Función objetivo para Optuna
            def objective(trial):
                try:
                    print(f"\n🔍 Trial {trial.number + 1}/{n_trials}")
                    print("-"*30)
                    
                    # Obtener parámetros sugeridos
                    params = self._create_model_params(trial, X, y)
                    print("Parámetros a probar:", {k: round(v, 4) if isinstance(v, float) else v 
                                                for k, v in params.items() if k != 'verbose'})
                    
                    # Configurar validación cruzada
                    cv = StratifiedKFold(
                        n_splits=n_splits,
                        shuffle=True,
                        random_state=self.config.get('model', {}).get('random_state', 42)
                    )
                    
                    # Métricas a calcular
                    metrics = {
                        'accuracy': [],
                        'precision': [],
                        'recall': [],
                        'f1': [],
                        'roc_auc': [],
                        'average_precision': []
                    }
                    
                    print(f"🏃 Ejecutando validación cruzada ({n_splits} folds)...")
                    
                    # Validación cruzada
                    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
                        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                        
                        # Crear datasets de LightGBM
                        train_data = lgb.Dataset(X_train, label=y_train)
                        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                        
                        # Entrenar modelo
                        model = lgb.train(
                            params=params,
                            train_set=train_data,
                            valid_sets=[train_data, valid_data],
                            valid_names=['train', 'valid'],
                            num_boost_round=self.config.get('model', {}).get('num_boost_round', 1000),
                            early_stopping_rounds=self.config.get('model', {}).get('early_stopping_rounds', 50),
                            verbose_eval=50  # Mostrar progreso cada 50 iteraciones
                        )
                        
                        # Hacer predicciones
                        y_pred = model.predict(X_val)
                        y_pred_binary = (y_pred > 0.5).astype(int)
                        
                        # Calcular métricas
                        metrics['accuracy'].append(accuracy_score(y_val, y_pred_binary))
                        metrics['precision'].append(precision_score(y_val, y_pred_binary, zero_division=0))
                        metrics['recall'].append(recall_score(y_val, y_pred_binary, zero_division=0))
                        metrics['f1'].append(f1_score(y_val, y_pred_binary, zero_division=0))
                        metrics['roc_auc'].append(roc_auc_score(y_val, y_pred))
                        metrics['average_precision'].append(average_precision_score(y_val, y_pred))
                        
                        print(f"  Fold {fold}: ROC-AUC = {metrics['roc_auc'][-1]:.4f}, F1 = {metrics['f1'][-1]:.4f}")
                    
                    # Calcular promedio de métricas
                    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
                    
                    # Mostrar resumen de métricas
                    print("\n📊 Métricas promedio:")
                    for metric_name, value in avg_metrics.items():
                        print(f"  {metric_name}: {value:.4f}")
                    
                    # Guardar métricas en el trial
                    for metric_name, value in avg_metrics.items():
                        trial.set_user_attr(metric_name, value)
                    
                    # Devolver la métrica principal
                    main_metric = self.config.get('evaluation', {}).get('main_metric', 'roc_auc')
                    print(f"\n🏆 Métrica principal ({main_metric}): {avg_metrics[main_metric]:.4f}")
                    
                    return avg_metrics[main_metric]
                    
                except Exception as e:
                    print(f"\n❌ Error en el trial {trial.number + 1}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return float('-inf')
            
            # Ejecutar optimización
            print("\n🚀 Iniciando búsqueda de hiperparámetros...")
            study.optimize(
                objective, 
                n_trials=n_trials, 
                timeout=timeout,
                callbacks=[log_trial]  # Mostrar progreso cuando hay mejora
            )
            
            # Guardar resultados
            self.study = study
            self.best_params = study.best_params
            self.best_score = study.best_value
            
            # Mostrar resumen final detallado
            print("\n" + "="*80)
            print("🏆 OPTIMIZACIÓN FINALIZADA - RESUMEN FINAL")
            print("="*80)
            
            # Mostrar información del mejor trial
            best_trial = study.best_trial
            print(f"\n🔍 Mejor trial (N°{best_trial.number}):")
            print(f"   - Valor de la métrica: {best_trial.value:.4f}")
            print(f"   - Duración: {best_trial.duration:.2f} segundos")
            
            # Mostrar los mejores parámetros formateados
            print("\n⚙️  MEJORES PARÁMETROS ENCONTRADOS:")
            print("-" * 50)
            for param, value in sorted(best_trial.params.items()):
                print(f"{param:>25}: {value}")
            
            # Mostrar importancia de parámetros
            print("\n📊 IMPORTANCIA DE PARÁMETROS:")
            print("-" * 50)
            importance = optuna.importance.get_param_importances(study)
            for param, imp in importance.items():
                print(f"{param:>25}: {imp*100:5.1f}%")
            
            # Guardar resultados en un DataFrame
            self.trials_df = study.trials_dataframe()
            
            # Guardar resultados en disco
            self._save_optimization_results()
            
            # Guardar los mejores parámetros en un archivo de configuración
            best_params_path = os.path.join(self.config['paths']['models_dir'], 'best_params_config.json')
            with open(best_params_path, 'w') as f:
                json.dump({
                    'model_params': best_trial.params,
                    'best_score': best_trial.value,
                    'optimization_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'config_used': self.config
                }, f, indent=4)
            
            print("\n💾 RESULTADOS GUARDADOS EN:")
            print(f"- trials.csv: {os.path.join(self.config['paths']['models_dir'], 'trials.csv')}")
            print(f"- best_params.json: {os.path.join(self.config['paths']['models_dir'], 'best_params.json')}")
            print(f"- best_params_config.json: {best_params_path}")
            print("\n" + "="*80)
            
            return study.best_params
            
        except Exception as e:
            print(f"\n❌ Error durante la optimización: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Devolver los mejores parámetros hasta el momento si hay algún error
            if hasattr(self, 'study') and self.study.best_trial:
                return self.study.best_params
            else:
                raise
        
        # Mostrar información del mejor trial
        best_trial = study.best_trial
        print(f"\n🔍 Mejor trial (N°{best_trial.number}):")
        print(f"   - Valor de la métrica: {best_trial.value:.4f}")
        print(f"   - Duración: {best_trial.duration:.2f} segundos")
        
        # Mostrar los mejores parámetros formateados
        print("\n⚙️  MEJORES PARÁMETROS ENCONTRADOS:")
        print("-" * 50)
        for param, value in sorted(best_trial.params.items()):
            print(f"{param:>25}: {value}")
        
        # Mostrar importancia de parámetros
        print("\n📊 IMPORTANCIA DE PARÁMETROS:")
        print("-" * 50)
        importance = optuna.importance.get_param_importances(study)
        for param, imp in importance.items():
            print(f"{param:>25}: {imp*100:5.1f}%")
        
        # Guardar resultados en un DataFrame
        self.trials_df = study.trials_dataframe()
        
        # Guardar resultados en disco
        self._save_optimization_results()
        
        # Guardar los mejores parámetros en un archivo de configuración
        best_params_path = os.path.join(self.config['paths']['models_dir'], 'best_params_config.json')
        with open(best_params_path, 'w') as f:
            json.dump({
                'model_params': best_trial.params,
                'best_score': best_trial.value,
                'optimization_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'config_used': self.config
            }, f, indent=4)
        
        print("\n💾 RESULTADOS GUARDADOS EN:")
        print(f"- trials.csv: {os.path.join(self.config['paths']['models_dir'], 'trials.csv')}")
        print(f"- best_params.json: {os.path.join(self.config['paths']['models_dir'], 'best_params.json')}")
        print(f"- best_params_config.json: {best_params_path}")
        print("\n" + "="*80)
        # Devolver los mejores parámetros hasta el momento si hay algún error
        if hasattr(self, 'study') and self.study.best_trial:
            return self.study.best_params
        else:
            raise
    
    def _save_optimization_results(self):
        """Guarda los resultados de la optimización."""
        if not hasattr(self, 'study'):
            return
        
        # Convertir los trials a DataFrame
        trials_df = self.study.trials_dataframe()
        
        # Guardar resultados en CSV
        results_path = self.optuna_dir / f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trials_df.to_csv(results_path, index=False)
        
        # Guardar los mejores parámetros
        best_params_path = self.optuna_dir / 'best_params.yaml'
        with open(best_params_path, 'w') as f:
            yaml.dump(self.best_params, f)
        
        print(f"Resultados de optimización guardados en: {results_path}")
        print(f"Mejores parámetros guardados en: {best_params_path}")
    
    def get_best_model(self, X_train, y_train, X_val=None, y_val=None):
        """
        Entrena un modelo con los mejores parámetros encontrados.
        
        Args:
            X_train: Datos de entrenamiento.
            y_train: Etiquetas de entrenamiento.
            X_val: Datos de validación (opcional).
            y_val: Etiquetas de validación (opcional).
            
        Returns:
            Modelo entrenado con los mejores parámetros.
        """
        if not hasattr(self, 'best_params'):
            raise ValueError("No se han encontrado parámetros optimizados. Ejecute optimize() primero.")
        
        # Crear datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(valid_data)
            valid_names.append('valid')
        
        # Entrenar modelo final
        model = lgb.train(
            params=self.best_params,
            train_set=train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            num_boost_round=self.config.get('model', {}).get('num_boost_round', 1000),
            early_stopping_rounds=self.config.get('model', {}).get('early_stopping_rounds', 50),
            verbose_eval=self.config.get('model', {}).get('verbose_eval', 50)
        )
        
        return model
    
    def get_best_params(self) -> Dict[str, Any]:
        """
        Devuelve los mejores parámetros encontrados.
        
        Returns:
            Diccionario con los mejores parámetros.
        """
        if not hasattr(self, 'best_params') and hasattr(self, 'study') and self.study.best_trial:
            self.best_params = self.study.best_params
            
        if not hasattr(self, 'best_params'):
            raise ValueError("La optimización no se ha ejecutado correctamente. No se encontraron parámetros óptimos.")
            
        return self.best_params
