# Interfaz didáctica para redes neuronales

## Objetivo acordado

Permitir que una persona sin programar defina una red neuronal, configure su
arquitectura, seleccione datos compatibles, entrene, evalúe y pruebe el modelo.
La aplicación, sus dependencias, la preparación de datos, el entrenamiento y la
inferencia deben ejecutarse dentro de Docker. Python mínimo: 3.12.

## Experiencia de uso

1. Crear o abrir un proyecto y elegir la tarea y el tipo de red.
2. Elegir un dataset de ejemplo o importar datos propios; mostrar una vista previa,
   etiquetas, dimensiones, cantidad de muestras y problemas encontrados.
3. Construir la arquitectura mediante controles visuales, con explicaciones de
   cada capa y una vista de las dimensiones de entrada y salida.
4. Validar la combinación de tarea, datos, preprocesamiento, capas, salida y pérdida.
   Los errores deben explicar cómo corregir la configuración antes de entrenar.
5. Configurar entrenamiento: particiones, semilla, épocas, tamaño de lote,
   optimizador y tasa de aprendizaje; ofrecer valores iniciales explicados.
6. Entrenar con progreso, gráficas de pérdida y métricas de entrenamiento y
   validación, y posibilidad de cancelar sin bloquear la interfaz.
7. Evaluar con un conjunto de prueba separado y métricas adecuadas a la tarea.
8. Cargar una entrada nueva y mostrar la predicción y su interpretación.
9. Guardar y reabrir el proyecto, los modelos y los resultados.

## Contenedores y dependencias

- El huésped aporta Linux, Docker y los drivers compatibles. El acceso a GPU
  requiere integración con Docker; NVIDIA necesita NVIDIA Container Toolkit.
- Instalar automáticamente las dependencias de la aplicación al construir las
  imágenes, con versiones controladas; el usuario final no debe ejecutar pip.
- Proponer imágenes o perfiles por backend (CPU, NVIDIA, AMD e Intel), con una
  matriz de compatibilidad verificada. No asumir que una imagen CUDA funciona
  con las otras marcas ni prometer soporte de todas las GPU de un fabricante.
- Verificar el backend desde el framework dentro del contenedor, incluyendo una
  operación real. La detección PCI actual solo es una pista de hardware presente.
- Si la GPU falla, explicar el motivo y ofrecer CPU cuando la tarea sea compatible.
- Descargar datasets desde la aplicación dentro del contenedor y conservarlos
  en almacenamiento persistente. Explicar origen, licencia y tamaño antes de descargar.
- Mantener datasets, proyectos y resultados en volúmenes para sobrevivir a la
  recreación del contenedor. Tras preparar imágenes y datos, permitir trabajo
  sin conexión en los flujos que no dependan de servicios externos.
- No instalar herramientas Python ni servicios auxiliares en el huésped para
  completar funciones de la aplicación.

## Separación propuesta

- Interfaz web en el navegador, acordada con el usuario: editor de arquitectura,
  importación de datos, progreso, resultados y pruebas. El servidor y el motor
  se ejecutan en Docker; el navegador es el punto de acceso del huésped.
- Dominio: configuración de proyectos y reglas de compatibilidad independientes
  de la interfaz; traducir configuraciones visuales a modelos sin ejecutar código
  arbitrario escrito por el usuario.
- Motor de ejecución: construir, entrenar, evaluar y aplicar modelos mediante
  PyTorch. La primera interfaz se implementa con Streamlit, en Python.
- Trabajos: ejecutar entrenamiento fuera del proceso que atiende la interfaz,
  registrar estados, errores y cancelación, y limitar concurrencia según recursos.
- Persistencia: guardar configuración, semilla, versiones, identificación del
  dataset, particiones, preprocesamiento, clases, pesos y métricas por ejecución.
  Usar en inferencia exactamente el preprocesamiento del entrenamiento.

## Primera entrega propuesta

Construir un recorrido completo de clasificación de imágenes con una CNN y un
dataset didáctico. El ejemplo existente de Fashion-MNIST puede servir como
referencia, pero debe adaptarse al contrato de Python >=3.12 y a la ejecución
íntegra dentro de Docker; sus instrucciones actuales usan tareas en el huésped.

La entrega debe permitir configurar capas compatibles, entrenar, consultar
métricas, guardar el modelo, reiniciar el contenedor y predecir una imagen con
el modelo guardado. Después se ampliarán tipos de redes y formatos de datos.

## Decisiones pendientes

- Tipos de red y tareas prioritarios para la primera versión.
- Formatos de datasets propios que se admitirán inicialmente.
- Hardware objetivo para verificar el primer backend acelerado.
- Framework y tecnologías de interfaz, tras concretar lo anterior.

## Referencias de infraestructura

- Docker Engine: https://docs.docker.com/engine/install/
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- Contenedores ROCm: https://rocm.docs.amd.com/en/latest/install/docker-containers.html

## Incremento implementado

Se incorpora primero clasificación tabular con MLP, Iris y CSV numéricos para
validar el recorrido completo con una imagen CPU. La CNN permanece como siguiente
extensión propuesta. El entrenamiento inicial es síncrono: trabajos en segundo
plano, cancelación y perfiles GPU están pendientes.

## Extensión CNN implementada

Se integra una CNN TensorFlow/Keras basada en `examples/red-neuronal-convolucional`.
El catálogo seleccionable contiene Fashion-MNIST y MNIST, ambos gratuitos y
compatibles con imágenes de 28×28 en gris. La arquitectura, entrenamiento,
evaluación y predicción desde imágenes se controlan desde la interfaz Python.
Los datasets se descargan desde el contenedor y se conservan en el volumen.
El perfil Docker sigue siendo CPU; aceleración GPU y cancelación siguen pendientes.
