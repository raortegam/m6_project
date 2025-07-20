from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from typing import List, Dict, Any
import pandas as pd

app = FastAPI(title="API de Predicción", 
             description="API para realizar predicciones con el modelo de machine learning",
             version="1.0.0")

# Cargar el modelo y preprocesador
model = None
preprocessor = None

def load_model():
    """Carga el modelo y el preprocesador desde archivos .pkl"""
    global model, preprocessor
    try:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_20250717_214252.pkl')
        preprocessor_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'preprocessor_20250717_214053.pkl')
        
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        print("Modelo y preprocesador cargados exitosamente")
        print("Modelo cargado:", model_path)
        print("Preprocesador cargado:", preprocessor_path)
        print("Tipo de modelo cargado:", type(model).__name__)
        print("Columnas esperadas por el preprocesador:", preprocessor.get_feature_names_out())
    except Exception as e:
        print(f"Error al cargar el modelo o preprocesador: {str(e)}")
        raise

# Cargar el modelo al iniciar
load_model()

class PredictionInput(BaseModel):
    features: List[Dict[str, Any]]

@app.get("/")
async def root():
    return {"message": "Bienvenido a la API de predicción"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

def predict_booster(booster, X):
    """Función para hacer predicciones con modelos Booster de LightGBM"""
    import lightgbm as lgb
    if isinstance(booster, lgb.Booster):
        # Si X es un array denso o una matriz dispersa, convertirlo a formato esperado
        if hasattr(X, 'toarray'):  # Para matrices dispersas
            X = X.toarray()
        elif hasattr(X, 'values'):  # Para DataFrames
            X = X.values
        # Hacer predicción directamente con el Booster
        predictions = booster.predict(X)
        # Para clasificación binaria, devolver ambas clases
        if predictions.ndim == 1 and len(np.unique(predictions)) <= 2:
            return predictions, np.column_stack((1-predictions, predictions))
        return predictions, None
    return None, None

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        # Convertir la entrada a DataFrame para el preprocesamiento
        import pandas as pd
        df = pd.DataFrame(input_data.features)
        
        # Guardar customerID si existe
        customer_ids = []
        if 'customerID' in df.columns:
            customer_ids = df['customerID'].tolist()
            df = df.drop(columns=['customerID'])
        else:
            # Si no hay customerID, generar uno ficticio
            customer_ids = [f"cust_{i}" for i in range(len(df))]
        
        # Añadir customerID temporalmente si es necesario
        df['customerID'] = customer_ids
        
        # Verificar las columnas esperadas por el preprocesador
        expected_columns = preprocessor.get_feature_names_out()
        print("Columnas esperadas por el preprocesador:", expected_columns)
        print("Columnas en los datos de entrada:", df.columns.tolist())
        
        # Asegurarse de que todas las columnas esperadas estén presentes
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            print(f"Añadiendo columnas faltantes con valores nulos: {missing_columns}")
            for col in missing_columns:
                if col != 'customerID':  # Ya manejamos customerID
                    df[col] = None
        
        # Aplicar preprocesamiento
        X = preprocessor.transform(df)
        print("Tipo de datos después del preprocesamiento:", type(X))
        
        # Realizar predicción
        try:
            # Primero intentar con predict_proba (para modelos scikit-learn)
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
        except (AttributeError, TypeError):
            # Si falla, usar el método para LightGBM Booster
            print("Usando predict_booster para modelo LightGBM")
            predictions, proba = predict_booster(model, X)
            if proba is not None:
                probabilities = proba
            else:
                # Si no hay probabilidades, crear un array de ceros
                probabilities = [[0.0, 0.0] for _ in range(len(predictions))]
        
        # Preparar la respuesta
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                "customer_id": customer_ids[i] if i < len(customer_ids) else f"cust_{i}",
                "prediction": float(pred),
                "probability": float(prob[1]) if isinstance(prob, (list, np.ndarray)) else float(prob)
            })
        
        return {"results": results}
        
    except Exception as e:
        import traceback
        error_detail = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_detail)  # Imprimir el error en la consola del servidor
        raise HTTPException(status_code=400, detail=str(e))  # Enviar solo el mensaje de error al cliente

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
