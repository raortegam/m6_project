# Reporte del Modelo Baseline

Este documento contiene los resultados del modelo baseline.

## Descripción del modelo

El modelo baseline `es el primer modelo construido y se utiliza para establecer una línea base para el rendimiento de los modelos posteriores.` que consideramos como primer modelo es un LightGBM (Light Gradient Boosting Machine) que consideramos como una opción excelente para tu escenario porque puede manejar las variables categóricas de forma eficiente y directa, es extremadamente rápido y preciso para problemas de clasificación binaria, y puede capturar patrones complejos en los datos que son comunes cuando la mayoría de las características son categóricas.

## Variables de entrada

* `customerID`
* `gender`
* `SeniorCitizen`
* `Partner`
* `Dependents`
* `tenure`
* `PhoneService`
* `MultipleLines`
* `InternetService`
* `OnlineSecurity`
* `OnlineBackup`
* `DeviceProtection`
* `TechSupport`
* `StreamingTV`
* `StreamingMovies`
* `Contract`
* `PaperlessBilling`
* `PaymentMethod`
* `MonthlyCharges`
* `TotalCharges`

## Variable objetivo

* `Churn`

## Evaluación del modelo

### Métricas de evaluación

- Accuracy (Exactitud)
La proporción de todas las predicciones correctas sobre el total. Te dice qué tan a menudo el modelo acertó en general. Ten cuidado: puede engañar si una clase es mucho más común que la otra.

- Precision (Precisión del Positivo)
Responde: "De todas las veces que mi modelo dijo que algo era positivo, ¿cuántas acertó de verdad?" Es clave cuando el costo de un Falso Positivo es muy alto.

- Recall (Sensibilidad o Exhaustividad)
Responde: "De todas las cosas que realmente eran positivas, ¿cuántas logró encontrar mi modelo?" Es vital cuando el costo de perder un caso positivo real (Falso Negativo) es muy alto.

- F1-Score
Es un equilibrio entre Precision y Recall. Te da una sola puntuación que resume qué tan bien el modelo balancea estos dos aspectos. Útil especialmente con datos desequilibrados.

- ROC AUC (Área Bajo la Curva ROC)
Mide la capacidad general del modelo para diferenciar entre la clase positiva y la negativa, sin importar el umbral que uses. Un valor más alto significa mejor distinción. Es robusta al desequilibrio de clases.

- Average Precision (AP) / PR AUC (Área Bajo la Curva Precision-Recall)
Es el área bajo la curva Precision-Recall. Es mucho más informativa que ROC AUC cuando la clase positiva es minoritaria (tu dataset está desequilibrado). Te da una visión más clara del rendimiento del modelo para encontrar esa clase rara.

### Resultados de evaluación

| Métrica           | Valor     |
| :---------------- | :-------- |
| **Accuracy** | 0.7029    |
| **Precision** | 0.4627    |
| **Recall** | 0.7299    |
| **F1-Score** | 0.5664    |
| **ROC AUC** | 0.7856    |
| **Average Precision** | 0.5412    |

## Análisis de los resultados

El modelo muestra una accuracy general del 70.3%, lo que indica que acierta en la mayoría de sus predicciones. Sin embargo, al observar más de cerca, vemos una precision del 46.3%. Esto significa que cuando tu modelo predice algo como positivo, se equivoca bastante a menudo, generando una cantidad significativa de Falsos Positivos. Como las consecuencias de una predicción positiva incorrecta no son altas, este no es un punto crucial a mejorar. A menos que se consideren costos elevados, por ejemplo aosciados a campañas de retencion.

Por otro lado, el recall del 73% es bastante sólido. Esto indica que tu modelo es muy bueno para identificar la mayoría de los casos positivos reales, lo cual es excelente si el costo de "perder" un caso positivo (un Falso Negativo) es elevado. El F1-score de 0.57 busca un equilibrio entre estas dos métricas, y su valor refleja el buen recall, aunque también está limitado por la baja precision.

Finalmente, el ROC AUC de 0.79 sugiere que tu modelo tiene una buena capacidad para discriminar entre las dos clases, es decir, puede diferenciar bastante bien entre lo positivo y lo negativo. La Average Precision de 0.54 es una métrica más realista para datasets desequilibrados y, aunque es moderada, complementa la visión de que el modelo tiene potencial, pero necesita ser más preciso en sus predicciones positivas para ser verdaderamente robusto. 

## Conclusiones

`Conclusiones generales sobre el rendimiento del modelo baseline y posibles áreas de mejora.`
En conclusion el modelo que es bueno encontrando los positivos, pero necesita ser más selectivo para evitar clasificar erróneamente los negativos.

`## Referencias`

`Lista de referencias utilizadas para construir el modelo baseline y evaluar su rendimiento.`

`Espero que te sea útil esta plantilla. Recuerda que puedes adaptarla a las necesidades específicas de tu proyecto.`
