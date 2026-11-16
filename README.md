# NN2GUI — Laboratorio de redes neuronales

Aplicación didáctica con interfaz en el navegador, escrita en Python y ejecutada
íntegramente en Docker. El usuario final configura y utiliza redes sin programar.

## Iniciar la aplicación

Requisitos del huésped: Linux, Docker Engine y Docker Compose. El perfil predeterminado
usa CPU; no requiere GPU, Python ni pip instalados en el huésped.

Desde esta carpeta:

```bash
docker compose up --build -d
```

Abre http://localhost:8501 en el navegador. La primera construcción descarga e
instala automáticamente Python 3.12, Streamlit, PyTorch y las demás dependencias
dentro de la imagen. Requiere Internet y espacio para las imágenes Docker.

Para detener:

```bash
docker compose down
```

Los datos y modelos permanecen en el volumen `proyectos`. No uses `down -v` si
quieres conservarlos. La aplicación se publica únicamente en localhost.

## Qué puedes hacer

1. Usar el dataset Iris incluido o cargar un CSV con entradas numéricas y etiquetas.
2. Elegir la columna objetivo y comprobar la compatibilidad de los datos.
3. Configurar una red densa (MLP) o una CNN TensorFlow basada en el ejemplo de `examples/`.
4. Ver las curvas de pérdida, exactitud de prueba y matriz de confusión.
5. Reabrir experimentos guardados y predecir con nuevas medidas.

Cada experimento conserva datos, pesos, normalización, clases, configuración,
particiones, semilla y métricas. La normalización se ajusta solo con entrenamiento.
La predicción recupera el modelo y el preprocesamiento desde disco.

## Alcance de esta primera versión

Clasificación tabular con MLP y clasificación de imágenes con CNN. No admite todavía
regresión ni redes recurrentes. Los trabajos CUDA se ejecutan en procesos separados;
los trabajos CPU siguen en la sesión. Todavía no hay cancelación ni cola de trabajos.
Los datasets cargados deben ser propios o tener permiso para usarlos. Un modelo
siempre elige entre las clases aprendidas; no detecta automáticamente datos ajenos.

El perfil CUDA permite seleccionar CPU o GPU NVIDIA por framework.
Además del driver y Docker, NVIDIA necesita NVIDIA Container Toolkit en el huésped.
No se instala software de entrenamiento en el huésped.

Consulta [la visión del proyecto](docs/vision-del-proyecto.md) para el flujo previsto.
El directorio `examples/` conserva un ejercicio independiente de CNN; su arquitectura y preprocesamiento sirven como base de la CNN de la interfaz.

## Comprobaciones dentro de Docker

```bash
docker compose run --rm -v "$PWD/tests:/app/tests:ro" laboratorio python -m unittest discover -s tests -p "test_[tac]*.py" -v
```

## Herramienta inicial de detección de hardware


Base de proyecto en **Python 3.12 o superior** que funciona **únicamente en Linux** y selecciona automáticamente el dispositivo de cómputo:

NVIDIA (CUDA) → AMD (ROCm) → Intel (oneAPI/XPU) → **CPU** si no hay GPU utilizable.

## Uso

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
proyecto            # muestra GPU detectadas y dispositivo elegido
pytest
```

Forzar dispositivo: `PROYECTO_DEVICE=cpu proyecto` (nvidia | amd | intel | cpu).

## Cómo se detecta

Sin dependencias: se leen las GPU PCI desde `/sys/bus/pci/devices` y se verifica el driver
(`/proc/driver/nvidia` o `nvidia-smi`, `/dev/kfd`, `/dev/dri/renderD*`).
Esta herramienta PCI es orientativa; los selectores de la interfaz comprueban CUDA
mediante operaciones reales en PyTorch o TensorFlow dentro del contenedor.

## CNN y catálogo de datasets públicos

Selecciona **Convolucional (CNN)** en la barra lateral. En la pestaña
**Entrenamiento**, elige manualmente el dataset:

| Dataset gratuito | Contenido | Entrada | Fuente |
| --- | --- | --- | --- |
| Fashion-MNIST | Diez categorías de prendas | 28×28, gris | [Zalando Research; MIT](https://github.com/zalandoresearch/fashion-mnist) |
| MNIST | Dígitos manuscritos del 0 al 9 | 28×28, gris | [Catálogo Keras](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data) |

La descarga ocurre solo al pulsar **Descargar y explorar dataset** o **Entrenar CNN
y guardar**, y se conserva en `/data/datasets`, dentro del volumen persistente.
Fashion-MNIST reutiliza el descargador del ejemplo con comprobaciones MD5; MNIST
usa el descargador oficial de Keras. Tras descargar, no se requiere conexión para
volver a entrenar con esos datos. La primera descarga requiere Internet.

La CNN reproduce por defecto dos bloques Conv2D (32 y 64 filtros 3×3, ReLU) y
MaxPooling2D, Flatten, Dense(64), Dropout(0.2) y Dense(10, softmax). Se pueden
configurar de uno a tres bloques, filtros, capa densa, dropout y entrenamiento.
Con tres bloques se usa padding `same` para conservar dimensiones compatibles.
Se reserva 1/6 de las muestras seleccionadas para validación y se evalúa sobre
el test oficial completo. La parada temprana recupera los mejores pesos.

Se guardan modelo `.keras`, configuración, identificación del dataset, índices de
particiones, métricas y una imagen del test para probar la recarga. Puedes subir
un PNG/JPEG propio: se convierte a gris, se redimensiona a 28×28 y se normaliza
igual que en el ejemplo. Solo reconocerá clases del dataset elegido; fotos de
otros objetos no son entradas fiables. La caché original conserva los datasets;
no la elimines si quieres volver a entrenar sin conexión.

Para actualizar una instalación anterior sin perder experimentos:

```bash
docker compose up --build -d
```

## Tres procesos separados en la interfaz

- Al abrir una sesión se presenta una introducción de doce páginas: portada,
  definición de red neuronal, neurona biológica, neurona artificial, funcionamiento
  de una neurona y de una red, tipos de redes y sus aplicaciones, activaciones,
  entrenamiento, evaluación y ejecución. La portada tiene un único botón
  **Comenzar**; usa **Anterior** y **Siguiente** para recorrer el contenido. Desde
  la barra lateral puedes volver a ella.
- **Arquitectura:** elegir el tipo de red en la barra lateral y configurar sus
  capas, neuronas, filtros, activaciones y dropout.
- **Entrenamiento:** seleccionar o cargar el dataset, ajustar épocas, lotes y tasa
  de aprendizaje, entrenar y consultar las métricas de los experimentos guardados.
- **Ejecución:** seleccionar cualquier modelo guardado (MLP o CNN), introducir
  medidas o cargar una imagen y obtener predicciones. No requiere volver a entrenar.

## Imágenes de la presentación

Las cinco gráficas de conceptos se extrajeron de la guía DOCX compartida para el
proyecto. La explicación de la neurona biológica, la neurona artificial y la red
completa incluye además estos recursos descargados de Wikimedia Commons:

- `neural-network-overview.png`: [Neural Network.svg](https://commons.wikimedia.org/wiki/File:Neural_Network.svg),
  de QuantuMechaniX8, CC0 1.0. Se usa la vista PNG del SVG.
- `biological-neuron.png`: [Neuron (PSF).png](<https://commons.wikimedia.org/wiki/File:Neuron_(PSF).png>),
  de Pearson Scott Foresman, dominio público.
- `artificial-neuron-detail.png`: [A Perceptron Neuron.png](https://commons.wikimedia.org/wiki/File:A_Perceptron_Neuron.png),
  de Arash Yoosefdoost, CC BY-SA 4.0.

Los cambios en Arquitectura se aplican al próximo entrenamiento. Ejecución usa
siempre la arquitectura y el preprocesamiento del modelo guardado, aunque cambies
el tipo de red o los controles de diseño. Los experimentos anteriores siguen siendo
compatibles con la nueva organización.

## Activaciones durante la ejecución

En **Ejecución**, activa **Visualizar activaciones en tiempo real** antes de pulsar
**Predecir** o **Clasificar imagen**. La visualización se actualiza al terminar cada
capa de la inferencia real, con una pausa didáctica opcional de 0 a 0.5 segundos.
Los valores corresponden a esa entrada y al modelo guardado; no se modifican pesos.

- MLP: entradas normalizadas, transformaciones lineales, activaciones y probabilidades.
- CNN: imagen normalizada, convoluciones, agrupaciones, aplanamiento, capa densa,
  dropout y salida. Dropout permanece desactivado durante inferencia.
- Al terminar, usa **Inspeccionar capa calculada** para recorrer los valores sin
  repetir la inferencia. Las capas espaciales muestran hasta ocho mapas y permiten
  consultar los valores originales de cualquier canal.

El diagrama CNN muestra la imagen de entrada, pilas de hasta cuatro mapas por capa,
una muestra del vector aplanado, hasta ocho neuronas densas y las probabilidades
de las diez clases. La red MLP muestra hasta diez neuronas por etapa. Los valores
completos se pueden consultar debajo. Azul representa valores positivos y naranja
negativos, con escala relativa por capa; el borde dorado resalta la etapa activa.
Las conexiones son esquemáticas y no representan pesos individuales. Activación no equivale a importancia causal ni atención.
La pausa es una ayuda visual y no mide el rendimiento del procesador. Cada nueva
predicción reemplaza la traza anterior; las trazas son temporales de la sesión.

## CPU o GPU NVIDIA con CUDA

El perfil CPU continúa funcionando con `docker compose up --build -d`.
Para habilitar CUDA (el mismo volumen conserva todos los modelos):

```bash
docker compose -f compose.yaml -f compose.cuda.yaml up --build -d
```

El huésped requiere driver NVIDIA funcionando y NVIDIA Container Toolkit configurado
para Docker. Comprueba `nvidia-smi` antes de iniciar el perfil CUDA. La imagen instala
las bibliotecas CUDA dentro del contenedor; no necesitas instalar Python ni CUDA
Toolkit en el huésped. La construcción CUDA descarga varios GB adicionales.

Selecciona **Dispositivo de cómputo** en Entrenamiento o Ejecución. Las elecciones
son independientes: un modelo entrenado con GPU se puede ejecutar en CPU y viceversa.
Cada GPU se identifica por nombre e índice. Solo se ofrecen dispositivos que superan
una operación real de comprobación del framework; si no hay GPU disponible se informa
el motivo y se conserva CPU. Un fallo durante un trabajo GPU muestra un error;
no se cambia silenciosamente de dispositivo. La configuración guardada registra
el dispositivo de entrenamiento.

PyTorch CUDA y TensorFlow CUDA se ejecutan en procesos y entornos independientes
dentro del mismo contenedor para evitar conflictos de bibliotecas. El progreso y
las activaciones siguen llegando a la interfaz durante el cálculo. La visualización
copia las activaciones a CPU y añade sobrecarga; desactívala para medir rendimiento.

Referencias: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
[TensorFlow CUDA](https://www.tensorflow.org/install/pip).

## Elegir imágenes del dataset para probar una CNN

En **Ejecución**, abre una CNN guardada y elige **Selección aleatoria de 5 000
imágenes** como origen. Pulsa **Extraer 5 000 imágenes al azar**. Se seleccionan
5 000 índices únicos del test oficial del mismo dataset del modelo, sin usar
imágenes de entrenamiento. La selección se guarda en el volumen de Docker;
puedes volver a abrirla o generar otra muestra aleatoria.

Explora la galería de 20 imágenes por página y pulsa **Elegir** debajo de una,
o introduce directamente un número entre 1 y 5 000. Luego pulsa **Clasificar
imagen**. La etiqueta real se muestra junto al resultado para comprobar el acierto.
La selección conserva los píxeles originales y usa el mismo preprocesamiento,
selector CPU/CUDA y visualización de activaciones de las demás entradas.
También siguen disponibles el ejemplo guardado y la subida de imágenes propias.
El muestreo guarda índices y semilla, sin duplicar 5 000 archivos PNG.

Usa esta galería para explorar errores, no para ajustar hiperparámetros según el
test; el entrenamiento debe guiarse por la validación.
