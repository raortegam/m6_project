# Usar una imagen base de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos de requisitos
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fastapi uvicorn pandas

# Copiar la aplicación
COPY ./api /app/api

# Copiar el modelo y preprocesador
COPY ./models/model_20250717_214053.pkl /app/api/models/
COPY ./models/preprocessor_20250717_214053.pkl /app/api/models/

# Puerto expuesto
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
