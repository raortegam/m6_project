# Definición de los datos

## Origen de los datos

- [ ] La fuente de los datos es **kaggle**, y fueron obtenidos mediante busqueda e indagación en el repositorio de datos en la plataforma.

## Especificación de los scripts para la carga de datos

- [ ] El script que permite realizar la descarga de los datos se encuentra dentro del reposiorio en la ruta: *scripts/data_acquisition/acquisition.py*
- [ ] En cuanto a la carga de los datos, esta se realiza haciendo uso del metodo de la libreria pandas para cargar archvios **.csv**

## Referencias a rutas o bases de datos origen y destino

- [ ] En este caso no hay un motor de bases de datos (MySQL, ORACLE, Cassandra, etc.) del cual se lean y escriban datos. 
- [ ] Actualmente y teniendo en cuenta que la base de datos es bastante ligera, los datos de entrada se encuentran almacenados en la ruta *src/database/WA_Fn-UseC_-Telco-Customer-Churn.csv*
- [ ] Finalmente en cuanto a los datos de salida como las predicciones obtenidas durante los entrenamientos seran almacenadas en la mis ruta *src/database/*.

### Rutas de origen de datos

- [ ] Ubicación de los archivos de origen de los datos:
  - [ ] src/database 
- [ ] La estructura de los archivos de origen de los datos, se compone de un unico archivo en formato *.CSV*, es decir, son datos estructurados en formato tabular donde las variables son principalmente categoricas.
      Solo hay 3 variables numéricas:
  - [ ] int64: ['SeniorCitizen', 'tenure']
  - [ ] float64: ['MonthlyCharges']
- [ ] Los procedimientos de transformación y limpieza de los datos son:
  - [ ] Validacion de valores faltantes
  - [ ] Validacion de valores duplicados
  - [ ] En cuanto a transformaciones como:
    - [ ] Recategorizar variables (No procede)
    - [ ] Aplicar funciones sobre las variables continuas para cambios de escala (No procede) 

### Base de datos de destino

- [ ] Ubicación de los archivos de destino de los datos:
  - [ ] src/database 
- [ ] La estructura de la base de datos de destino consta del ID del cliente, la probabilidad estimada y la categoria estimada.
