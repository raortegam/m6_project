# Reporte de Datos

Este documento contiene los resultados del análisis exploratorio de datos.

## Resumen general de los datos

En esta sección se presenta un resumen general de los datos. `Se describe el número total de observaciones, variables, el tipo de variables, la presencia de valores faltantes y la distribución de las variables.`
- La base de datos cuenta con 7043 registros
- Solamente 3 de las variables son numericas
- Casi la totalidad de las variables son categoricas con hasta 3 categorias
- Las variables estan muy orientadas a la cantidad, tipo y valor de los servicios que tiene contratados el cliente
- Demograficamente solo sabemos si es adulto mayor o no y su genero
- Una variable me habla de los metodos de pago.
- No hay presencia de valores faltantes.
- La categoria mayoritaria para churn es No con un 73.5% de los datos. No se considera que halla un gran desbalanceo en la variable churn.

## Resumen de calidad de los datos

En esta sección se presenta un resumen de la calidad de los datos. `Se describe la cantidad y porcentaje de valores faltantes, valores extremos, errores y duplicados. También se muestran las acciones tomadas para abordar estos problemas.`
- No hay presencia de valores faltantes.
- No hay presencia de valores extremos teniendo en cuenta que la mayoria de las variables son categoricas.
- Realizando la validacion de duplicidad, se encuientra que no hay presencia de duplicados.
- Realizando la validacion correspondiente no hay registros truncados o con errores
- No proceden acciones relevantes.

## Variable objetivo

En esta sección se describe la variable objetivo. Se muestra la distribución de la variable y se presentan gráficos que permiten entender mejor su comportamiento.

- Distribución de Churn:
 - No: 5174 (73.5%)
 - Yes: 1869 (26.5%)

Se generaron graficas, las cueles se encuentran en: m6_project/scripts/eda/eda.ipynb

## Variables individuales

En esta sección se presenta un análisis detallado de cada variable individual. `Se muestran estadísticas descriptivas, gráficos de distribución y de relación con la variable objetivo (si aplica). Además, se describen posibles transformaciones que se pueden aplicar a la variable.`

- Solo un 16% de los clientes son personas mayores.
- Hay muchos clientes nuevos (tenure bajo) y varios clientes antiguos. Dado que media > mediana, la distribución tiene una ligera cola derecha (asimetría positiva).
- MonthlyCharges Distribución: Sesgada hacia la izquierda (asimetría negativa).

Graficas y detalles se encuentran en: m6_project/scripts/eda/eda.ipynb

`## Ranking de variables`

`En esta sección se presenta un ranking de las variables más importantes para predecir la variable objetivo.` `Se utilizan técnicas como la correlación, el análisis de componentes principales (PCA) o la importancia de las variables en un modelo de aprendizaje automático.`

## Relación entre variables explicativas y variable objetivo

En esta sección se presenta un análisis de la relación entre las variables explicativas y la variable objetivo. Se utilizan gráficos como la matriz de correlación y el diagrama de dispersión para entender mejor la relación entre las variables. Además, se pueden utilizar técnicas como la regresión lineal para modelar la relación entre las variables.

El análisis de las variables categóricas en relación con el churn revela varios patrones importantes. Variables como gender, PhoneService y MultipleLines no muestran diferencias significativas en la tasa de abandono, por lo que parecen tener poca capacidad predictiva.
En contraste, tener pareja (Partner) o dependientes (Dependents) se asocia con una menor tasa de churn, lo que sugiere que clientes con responsabilidades familiares podrían tener mayor estabilidad en el servicio. 

El tipo de contrato es uno de los factores más determinantes: los contratos mensuales presentan una tasa de abandono muy superior (~43%) frente a los contratos anuales o bienales, donde el churn es inferior al 10%, lo que refleja un mayor compromiso por parte de los clientes con contratos más largos.

En cuanto al método de pago, los clientes que usan transferencias bancarias o tarjetas de crédito automáticas muestran una menor tasa de churn, mientras que quienes pagan por cheque electrónico tienen una tasa mucho más alta (~45%), posiblemente indicando menor fidelización. 

Tener servicios como OnlineSecurity, TechSupport, DeviceProtection y OnlineBackup también se relaciona con una menor propensión al abandono, sugiriendo que los servicios complementarios generan una mayor retención. Por el contrario, clientes con InternetService de fibra óptica y PaperlessBilling presentan tasas de churn por encima del promedio, lo que podría estar vinculado con un perfil más volátil o una mayor exposición a la competencia. 

Finalmente, los clientes que no contratan servicios de entretenimiento como StreamingTV y StreamingMovies también muestran una tasa mayor de abandono, posiblemente por menor integración con el ecosistema del proveedor. En conjunto, estos hallazgos permiten identificar perfiles de riesgo y factores clave para estrategias de retención.

Los histogramas segmentados por churn muestran que las variables numéricas tienen una relación clara con la probabilidad de abandono. En el caso de SeniorCitizen, aunque la mayoría de los clientes no son adultos mayores, se observa que quienes sí lo son presentan una proporción más alta de churn, lo que sugiere que la edad podría influir en la decisión de abandonar el servicio. 

La variable tenure evidencia una relación inversa muy marcada: los clientes con poca antigüedad (especialmente entre 0 y 10 meses) concentran la mayor parte de los abandonos, mientras que aquellos con más tiempo en la compañía casi no desertan, lo que la convierte en un fuerte predictor. 

En cuanto a MonthlyCharges, se observa que los clientes con cargos mensuales altos tienen tasas más elevadas de churn, posiblemente por insatisfacción con el precio o servicios contratados, mientras que los clientes con tarifas bajas presentan menor abandono.
Finalmente, TotalCharges, al estar relacionado con el tiempo y el valor del cliente, muestra que quienes han pagado menos (clientes recientes o de menor consumo) son mucho más propensos a irse, mientras que los clientes con cargos acumulados altos rara vez abandonan.

En conjunto, estos hallazgos refuerzan que tenure y TotalCharges son fuertes indicadores de retención, mientras que MonthlyCharges y SeniorCitizen pueden ayudar a identificar perfiles de mayor riesgo de churn.

