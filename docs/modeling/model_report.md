# Reporte del Modelo Final

## Resumen Ejecutivo

`En esta sección se presentará un resumen de los resultados obtenidos del modelo final. Es importante incluir los resultados de las métricas de evaluación y la interpretación de los mismos.`
El presente informe detalla el desarrollo y la evaluación de un modelo de clasificación binaria basado en LightGBM, optimizado mediante la librería Optuna para la búsqueda de hiperparámetros. El objetivo principal fue abordar un problema de clasificación (ej. predicción de abandono de clientes, fraude, etc., a definir en la sección siguiente), logrando un balance entre la capacidad de identificar casos positivos y mantener la precisión en dichas predicciones. Los resultados obtenidos muestran que el modelo final alcanza una Accuracy del 76.05%, una Precision del 54.76%, un Recall del 56.95%, un F1-Score de 55.83%, un ROC AUC de 80.22% y un Average Precision de 57.76%. Estas métricas reflejan una buena capacidad discriminativa general, aunque señalan áreas de mejora en la precisión de las predicciones positivas.

## Descripción del Problema

El problema central que este modelo busca resolver es la predicción del abandono de clientes (churn). En el contexto de una empresa de telecomunicaciones, identificar proactivamente los casos de clientes que abandonarán es crucial. Los objetivos principales son reducir la tasa de churn y mejorar la retención de clientes.

La justificación de este modelo radica en la necesidad de transformar grandes volúmenes de datos en inteligencia procesable, permitiendo a las unidades de negocio dentro de la empresa tomar decisiones proactivas y dirigidas. La identificación temprana de clientes en riesgo habilita la implementación de estrategias específicas para campañas de retención personalizadas optimizando recursos y maximizando el impacto empresarial.

## Descripción del Modelo

Para abordar el problema planteado, se desarrolló un modelo de clasificación binaria utilizando la librería LightGBM (Light Gradient Boosting Machine). LightGBM es un framework de gradient boosting basado en árboles de decisión, reconocido por su eficiencia, velocidad de entrenamiento y bajo consumo de memoria, lo que lo hace ideal para conjuntos de datos grandes.

La metodología para la construcción del modelo incluyó las siguientes fases:

- Preprocesamiento de Datos: Se realizó la limpieza y preparación de las variables existentes, asegurando su formato adecuado para el entrenamiento del modelo. Se mantuvieron las variables originales del dataset, sin exclusión de características.

- Manejo de Variables Categóricas: LightGBM tiene la capacidad de manejar variables categóricas de forma nativa, lo que simplifica el proceso de codificación para este tipo de características (que son la mayoría en el dataset).

- Optimización de Hiperparámetros con Optuna: La fase crítica de desarrollo fue la optimización de los hiperparámetros de LightGBM. Para ello, se empleó la librería Optuna, un framework de optimización que automatiza la búsqueda de los mejores conjuntos de hiperparámetros. Optuna explora eficientemente el espacio de parámetros para maximizar (o minimizar) una métrica objetivo (en este caso, se asume que se optimizó una métrica como ROC AUC o F1-Score) a través de un proceso de prueba y error estructurado, mejorando significativamente el rendimiento del modelo más allá de una configuración por defecto o manual.

El modelo final es el resultado de este proceso de ajuste fino, aprovechando las ventajas computacionales de LightGBM y la eficiencia de Optuna para encontrar una configuración de parámetros robusta que se desempeña de manera óptima en las métricas de evaluación clave.
  
## Evaluación del Modelo

La evaluación del modelo se realizó utilizando un conjunto estándar de métricas de clasificación binaria, las cuales proporcionan una visión integral de su rendimiento. A continuación, se presentan los resultados y su interpretación:

| Métrica           | Valor     |
| :---------------- | :-------- |
| **Accuracy** | 0.7605    |
| **Precision** | 0.5476    |
| **Recall** | 0.5695    |
| **F1-Score** | 0.5583    |
| **ROC AUC** | 0.8022    |
| **Average Precision** | 0.5776    |

El modelo muestra una accuracy del 76.05%, lo que indica que es correcto en aproximadamente tres de cada cuatro predicciones. Sin embargo, al profundizar, observamos una precision del 54.76%, lo que sugiere que casi la mitad de las veces que el modelo predice un caso positivo, se equivoca, generando bastantes falsos positivos. Por otro lado, el recall es del 56.95%, lo que significa que el modelo logra identificar poco más de la mitad de todos los casos que son realmente positivos, resultando en un número considerable de falsos negativos (casos positivos que el modelo no detecta).

El F1-score de 0.5583 refleja este balance moderado entre precisión y recall; al estar ambos valores relativamente cerca, el modelo no se inclina fuertemente hacia evitar un tipo de error sobre el otro, pero tampoco sobresale en ninguno. A pesar de esto, el ROC AUC de 0.8022 es bastante bueno, indicando que el modelo tiene una excelente capacidad general para distinguir entre las clases positiva y negativa. Finalmente, el Average Precision de 0.5776 es una métrica más reveladora en escenarios con desequilibrio de clases, y aunque es moderada, complementa la visión de que el modelo discrimina bien, pero tiene un margen significativo para mejorar en la precisión de sus predicciones positivas.

## Conclusiones y Recomendaciones

El modelo LightGBM, optimizado con Optuna, demuestra una excelente capacidad para diferenciar entre las clases, como lo indica su robusto ROC AUC del 80.22%. Esto sugiere que ha aprendido bien los patrones subyacentes en tus datos. Además, al mantener todas las variables originales, el modelo aprovecha la información completa disponible.

Sin embargo, a pesar de su buena capacidad de discriminación, el modelo actual presenta un equilibrio moderado entre Precision (54.76%) y Recall (56.95%), reflejado en un F1-Score de 55.83%. Esto significa que aún hay un margen considerable para reducir tanto los falsos positivos (cuando el modelo predice algo como positivo erróneamente) como los falsos negativos (cuando no detecta un caso positivo real).

Para mejorar el rendimiento se podria explorar:

- Ajustar el Umbral de Decisión: Evalúar los costos de los Falsos Positivos y Falsos Negativos en tu negocio. Si evitar falsos positivos es más crítico, podrías aumentar el umbral de probabilidad para clasificar como positivo. Si detectar la mayor cantidad de verdaderos positivos es lo primordial, podrías reducirlo.

- Ingeniería de Características: Explorar la creación de nuevas variables a partir de las existentes o la incorporación de nuevas fuentes de datos. Esto a menudo es clave para mejorar la precisión y el recall de un modelo.

- Análisis de Errores: Revisar los casos específicos donde el modelo se equivoca (Falsos Positivos y Falsos Negativos). Entender por qué falla en esos casos puede revelar patrones que el modelo no está capturando y guiar futuras mejoras.
   
`## Referencias`

`En esta sección se deben incluir las referencias bibliográficas y fuentes de información utilizadas en el desarrollo del modelo.`
