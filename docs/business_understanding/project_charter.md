# Project Charter - Entendimiento del Negocio

## Nombre del Proyecto

Modelo de Churn para Clientes de Telecomunicaciones

## Objetivo del Proyecto

Desarrollar un modelo predictivo de clasificación que identifique la probabilidad de que un cliente abandone el servicio de telecomunicaciones (churn), con el fin de anticipar la pérdida de clientes y permitir la implementación de estrategias de retención más efectivas. Este modelo contribuirá a reducir la tasa de cancelación de contratos, optimizando esfuerzos comerciales y de fidelización.

## Alcance del Proyecto

### Incluye:

- Análisis del dataset **Telco Customer Churn**, que contiene información demográfica, de servicios y de facturación de los clientes.
- Desarrollo de un modelo de clasificación para predecir la variable `Churn`.
- Evaluación del modelo con métricas como precisión, recall, F1 y ROC-AUC.
- Recomendaciones sobre acciones a tomar para los segmentos con mayor riesgo de churn.

### Excluye:

- Implementación del modelo en sistemas productivos reales.
- Costeo de campañas de retención basadas en los resultados del modelo.
- Integración con bases de datos externas o privadas.

## Metodología

Se utilizará una metodología CRISP-DM adaptada a proyectos ágiles. Se incluirán etapas de:
- Entendimiento del negocio y exploración de los datos.
- Limpieza y transformación del dataset.
- Entrenamiento de modelos de clasificación (árboles de decisión, random forest, gradient boosting).
- Evaluación comparativa de modelos.
- Interpretación de resultados y entrega de recomendaciones.

## Cronograma

| Etapa                                      | Duración Estimada | Fechas                     |
|-------------------------------------------|-------------------|----------------------------|
| Entendimiento del negocio y carga de datos| 1 semana          | del 1 de julio al 7 de julio |
| Preprocesamiento y análisis exploratorio  | 1 semana          | del 8 de julio al 14 de julio |
| Modelamiento y evaluación de modelos      | 1.5 semanas       | del 15 de julio al 24 de julio |
| Entrega de resultados y documentación     | 0.5 semanas       | del 25 de julio al 31 de julio |

## Equipo del Proyecto

- Richard Ortega – Científico de Datos Líder
- Mariana Beatriz Cruz Chú – Analista de Datos y Visualización
- Jeisson Andres Morales Hernandez – Ingeniero de Machine Learning

## Presupuesto

El proyecto contará con un presupuesto estimado de **USD 4,000**, destinado a:
- Infraestructura en la nube (computación y almacenamiento).
- Licencias de software y herramientas.
- Horas de trabajo dedicadas por el equipo técnico.

## Stakeholders

- Gerente de Retención de Clientes  
  Relación: Usuario principal del modelo para campañas de retención.  
  Expectativas: Recibir un sistema confiable para identificar clientes en riesgo y poder priorizar intervenciones.

- Director de Datos y Analítica  
  Relación: Responsable del alineamiento con la estrategia de datos de la empresa.  
  Expectativas: Uso eficiente de los recursos y generación de valor mediante ciencia de datos.

- Coordinadora de Marketing Digital  
  Relación: Potencial usuaria de los insights para campañas segmentadas.  
  Expectativas: Identificar perfiles propensos al abandono para personalizar mensajes.

## Aprobaciones

- Gerente de Innovación y Tecnología  
  Firma del aprobador: _______________________  
  Fecha de aprobación: 1 de julio de 2025
