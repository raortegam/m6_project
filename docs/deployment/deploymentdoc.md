# Despliegue de modelos

## Infraestructura

- **Nombre del modelo:** El modelo entrenado se identifica como `model_20250717_214053.pkl` y su preprocesador como `preprocessor_20250717_214053.pkl`.
- **Plataforma de despliegue:** La solución se basa en `Docker` para la contenerización, permitiendo su despliegue en cualquier entorno que soporte Docker Engine.
- **Requisitos técnicos:**
  * **Versión de Python:** Python 3.9.
    * **Bibliotecas:** Las librerías se instalan a partir del archivo `requirements.txt`. Además, se instalan `fastapi`, `uvicorn`, y `pandas`.
    * **Hardware:**
        * **CPU:** Se recomienda un mínimo de 2 vCPU para asegurar un rendimiento óptimo y capacidad de respuesta.
        * **RAM:** Mínimo 2 GB de RAM son necesarios; sin embargo, esto puede ajustarse en función del tamaño del modelo cargado y el volumen esperado de solicitudes de inferencia.
        * **Espacio en Disco:** Se requieren al menos 5 GB para almacenar la imagen de Docker, el código de la aplicación, los modelos y los archivos de logs.
    * **Software Adicional:** `Docker Engine` (versión 20.10.x o superior).
- **Requisitos de seguridad:**
      * **Autenticación:** Se espera implementar un mecanismo de autenticación (ej., claves API, tokens JWT o OAuth2) para los *endpoints* de la API para garantizar que solo usuarios o servicios autorizados puedan acceder al modelo.
    * **Encriptación de datos:** Se recomienda el uso de `HTTPS` para encriptar los datos en tránsito. Adicionalmente, se debe considerar el cifrado de datos en reposo si el modelo o los datos de entrada/salida se persisten.
    * **Acceso Mínimo (Least Privilege):** El usuario o servicio que ejecute el contenedor Docker debe operar con los permisos mínimos necesarios para su funcionamiento.
    * **Análisis de Vulnerabilidades:** Se recomienda realizar escaneos de seguridad periódicos en la imagen de Docker para identificar y mitigar posibles vulnerabilidades antes del despliegue en producción.
- **Diagrama de arquitectura:**

  ```mermaid
    graph TD
        A[Cliente / Aplicación Externa] -->|Solicitud HTTP GET/POST| B(Balanceador de Carga / API Gateway)
        B --> C[Servidor / Instancia de Nube]
        C --> D(Docker Engine)
        D --> E[Contenedor Docker]
        E --> F[Aplicación FastAPI / Uvicorn]
        F --> G[Carga de Modelo & Preprocesador .pkl]
        G --> H{Modelo ML entrenado}
        H --> I[Respuesta de Predicción]
        I --> F
        F --> E
        E --> D
        D --> C
        C --> B
        B --> A
    ```
    *Descripción:* La arquitectura propuesta despliega el modelo de Machine Learning como una `API RESTful` contenida en un `contenedor Docker`. Las solicitudes de predicción se originan desde un `cliente` (ej., otra aplicación web, móvil, o un script) y se dirigen a un `balanceador de carga` (opcional en desarrollo, crucial en producción) que distribuye el tráfico al `servidor` anfitrión. El `Docker Engine` en el servidor ejecuta el `contenedor Docker` que alberga la `aplicación FastAPI` servida por `Uvicorn`. Esta aplicación es responsable de cargar el `modelo` y el `preprocesador` serializados (archivos `.pkl`) y de realizar la `inferencia`. La `predicción` resultante es devuelta al cliente a través de la misma ruta. Se recomienda monitorear activamente el rendimiento del modelo y la infraestructura para detectar `drifting` y asegurar un mantenimiento proactivo.


## Código de despliegue

- **Archivo principal:** El archivo principal que contiene la lógica de la API y la interacción con el modelo es `api/main.py`. Este archivo es el que `Uvicorn` está configurado para ejecutar al iniciar el contenedor.

- **Rutas de acceso a los archivos:**
    * `./Dockerfile`: Archivo que contiene las instrucciones para construir la imagen Docker del proyecto.
    * `./requirements.txt`: Archivo que especifica las dependencias de Python del proyecto.
    * `./api/`: Directorio que contiene todo el código fuente de la aplicación FastAPI.
        * `./api/main.py`: El script principal de la aplicación FastAPI.
    * `./api/models/`: Directorio donde se almacenan los modelos y preprocesadores serializados dentro del contenedor.
        * `./models/model_20250717_214053.pkl`: El archivo del modelo de Machine Learning entrenado.
        * `./models/preprocessor_20250717_214053.pkl`: El archivo del objeto preprocesador de datos entrenado.

- **Variables de entorno:**
    Actualmente, el `Dockerfile` no expone variables de entorno específicas. Sin embargo, en un entorno de producción, se podrían considerar variables como:
    * `APP_PORT`: Para configurar dinámicamente el puerto de la aplicación.
    * `LOG_LEVEL`: Para ajustar el nivel de detalle de los registros.
    * `DATABASE_URL`: Si el modelo necesitara interactuar con una base de datos externa.


## Documentación del despliegue

- **Instrucciones de instalación:**
    1.  **Instalar Docker:** Asegúrate de tener `Docker Engine` instalado en tu sistema. Las instrucciones detalladas para tu sistema operativo se encuentran en la [documentación oficial de Docker](https://docs.docker.com/get-docker/).
    2.  **Clonar el repositorio:** Abre tu terminal o línea de comandos y clona el repositorio de GitHub de tu proyecto. (Reemplaza `tu-usuario/tu-repositorio` con la URL real de tu repositorio):
        ```bash
        git clone [https://github.com/tu-usuario/tu-repositorio.git](https://github.com/tu-usuario/tu-repositorio.git)
        ```
    3.  **Navegar al directorio del proyecto:** Accede al directorio raíz de tu proyecto clonado:
        ```bash
        cd tu-repositorio
        ```
    4.  **Construir la imagen Docker:** Desde el directorio raíz del proyecto (donde se encuentra el `Dockerfile`), ejecuta el siguiente comando para construir la imagen Docker. Dale un nombre significativo a tu imagen (ej., `modelo-ml-api`):
        ```bash
        docker build -t modelo-ml-api .
        ```
        Este proceso puede tardar unos minutos, ya que Docker descarga la imagen base e instala todas las dependencias.

- **Instrucciones de configuración:**
    1.  **Configuración de Puerto:** La aplicación FastAPI dentro del contenedor está configurada para escuchar en el puerto `8000`. Al ejecutar el contenedor, deberás mapear este puerto a un puerto en tu máquina anfitriona.
    2.  **Manejo de Variables de Entorno:** Si tu aplicación FastAPI necesitara variables de entorno (por ejemplo, para claves API, configuración de bases de datos, etc.), estas deben pasarse al contenedor durante su ejecución utilizando el flag `-e` en el comando `docker run`.
    3.  **Persistencia de Datos (Opcional):** Si tu modelo requiere leer o escribir datos en volúmenes persistentes, se pueden configurar volúmenes de Docker mediante el flag `-v` en el comando `docker run`.

- **Instrucciones de uso:**
    1.  **Ejecutar el contenedor Docker:** Para iniciar la API del modelo en segundo plano (`-d` para *detached mode*), mapeando el puerto `8000` del contenedor al puerto `8000` de tu máquina local, ejecuta:
        ```bash
        docker run -d -p 8000:8000 --name mi-modelo-api modelo-ml-api
        ```
        Donde `--name mi-modelo-api` asigna un nombre fácil de recordar a tu contenedor.
    2.  **Verificar el estado del contenedor:** Para asegurarte de que el contenedor se está ejecutando correctamente, puedes usar:
        ```bash
        docker ps
        ```
    3.  **Acceder a la API:** Una vez que el contenedor esté en ejecución, la API estará accesible en `http://localhost:8000` (o la IP del servidor si no es `localhost`).
        * Puedes acceder a la documentación interactiva de la API (Swagger UI) en: `http://localhost:8000/docs`
        * También puedes acceder a la documentación alternativa de Redoc en: `http://localhost:8000/redoc`
    4.  **Realizar predicciones:** Para interactuar con el modelo y obtener predicciones, envía una solicitud `HTTP POST` al *endpoint* de inferencia (`/predict`). El cuerpo de la solicitud debe ser un objeto JSON con los datos de entrada esperados por tu modelo.
        * **Ejemplo de solicitud `curl`:**
            ```bash
            curl -X POST "http://localhost:8000/predict" \
                 -H "Content-Type: application/json" \
                 -d '{
                       "feature_numerica_1": 123.45,
                       "feature_categorica_1": "valor_A",
                       "feature_numerica_2": 67.89
                     }'
            ```
            *(**Nota:** Ajusta los nombres de las `features` y los tipos de datos al esquema de entrada real de tu modelo.)*

- **Instrucciones de mantenimiento:**
    * **Monitoreo del Rendimiento del Modelo:**
        * Es crucial monitorear continuamente el rendimiento del modelo en producción para detectar `model drift` (degradación del rendimiento debido a cambios en la distribución de los datos de entrada o en la relación entre los datos y la etiqueta).
        * Métricas clave a monitorear incluyen la precisión del modelo, la latencia de las predicciones, y la tasa de errores de la API.
        * Herramientas como `Prometheus` y `Grafana` pueden ser utilizadas para recolectar, almacenar y visualizar estas métricas en tiempo real.
    * **Actualización del Modelo:**
        * Cuando se entrena un nuevo modelo con un mejor rendimiento o nuevos datos, los archivos `.pkl` actualizados deben copiarse en la ubicación correspondiente (`./models/`) en tu repositorio.
        * Después de actualizar los archivos, se debe reconstruir la imagen Docker (`docker build`) y redeployar el contenedor para que el nuevo modelo entre en vigor.
    * **Actualización de Dependencias:**
        * Mantén el archivo `requirements.txt` actualizado con las versiones más recientes y seguras de las librerías.
        * Reconstruye la imagen Docker periódicamente para incorporar actualizaciones de seguridad de la imagen base de Python y de las librerías instaladas.
    * **Gestión de Registros (Logs):**
        * Configura un sistema centralizado de logs (ej., `ELK Stack` (Elasticsearch, Logstash, Kibana), `Splunk`, o servicios de logging de la nube) para recolectar y analizar los registros generados por la aplicación FastAPI. Esto es vital para la depuración, el análisis de problemas y la auditoría.
    * **Optimización de Costos y Escalabilidad:**
        * Evalúa regularmente el consumo de recursos de la infraestructura desplegada (CPU, RAM). Ajusta el tamaño de las instancias del servidor o los límites de recursos de Docker para optimizar los costos.
        * Para manejar cargas de trabajo variables, se podría implementar estrategias de autoescalado a nivel de contenedores (ej., con Kubernetes) o de la plataforma (ej., Google Cloud Run, AWS Fargate), lo que permite escalar recursos solo cuando son necesarios y reducir costos durante períodos de baja demanda.
