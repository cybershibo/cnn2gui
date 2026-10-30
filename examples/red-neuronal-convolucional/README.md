# Armando — Caso práctico de red neuronal: clasificador de prendas

Repositorio **independiente** para el ejercicio de Tópicos Selectos. Incluye una red convolucional construida en Python, la descarga explícita de un conjunto de datos abierto, entrenamiento y evaluación, un PNG de entrada con su predicción, y un informe que se completa con las mediciones de la ejecución. Pensado para la laptop XENIA XPG 15KC con Ubuntu 26.04 y RTX 3070; también puede ejecutarse en CPU.

**Ruta corta:** desde esta carpeta ejecuta `python3 -m caso download`, después `docker build -t armando-cnn:tf215 .` y finalmente el comando `train` de la sección 5. La descarga requiere solo Python de Ubuntu; el entrenamiento usa Docker y lee los archivos descargados desde la carpeta `data/fashion-mnist/`.

## 1. Caso elegido y justificación

**Problema:** asignar a una imagen de una prenda una de diez categorías: camiseta, pantalón, suéter, vestido, abrigo, sandalia, camisa, tenis, bolsa o botín. **Entrada:** una imagen de 28×28 píxeles en escala de grises. **Salida:** diez probabilidades y la categoría con la probabilidad mayor. Es un ejemplo de clasificación visual verificable: se puede ver la prenda, conocer su etiqueta real y comparar la respuesta de la red.

Este caso permite estudiar adquisición, separación de datos, arquitectura, entrenamiento, errores y tiempos en un equipo real. Puede inspirar un clasificador de productos, pero el modelo entrenado aquí **solo conoce imágenes con el estilo de Fashion-MNIST**: no se debe presentar su exactitud como la de una cámara o una línea de producción.

## 2. Tipo de red y entrenamiento

Se construye una **red neuronal convolucional (CNN)** con aprendizaje **supervisado**: cada imagen del entrenamiento tiene una etiqueta entera del 0 al 9. Una convolución aplica filtros a zonas pequeñas de la imagen y aprende patrones como bordes; una agrupación reduce las dimensiones; las capas densas combinan esos rasgos. `Dropout(0.2)` desactiva parte de las unidades durante el entrenamiento para dificultar que la red memorice ejemplos. La última capa `softmax` da una distribución de probabilidades.

| Capas en orden | Salida | Papel |
| --- | --- | --- |
| Entrada 28×28×1 | 28×28×1 | Intensidad normalizada de cada píxel |
| Convolución 32 filtros 3×3 + ReLU; agrupación 2×2 | 13×13×32 | Extraer rasgos locales |
| Convolución 64 filtros 3×3 + ReLU; agrupación 2×2 | 5×5×64 | Combinar rasgos espaciales |
| Aplanar; densa 64 + ReLU; `Dropout(0.2)` | 64 | Clasificar con regularización |
| Densa 10 + `softmax` | 10 | Probabilidad de cada clase |

La función de pérdida es `sparse_categorical_crossentropy`, adecuada para diez clases excluyentes codificadas con enteros. El optimizador **Adam** cambia los pesos para reducir esa pérdida mediante retropropagación. `accuracy` es el porcentaje de imágenes cuya clase predicha coincide con la etiqueta. `model.fit` usa lotes de 128 imágenes y hasta diez épocas; `EarlyStopping` observa la pérdida de validación y restaura los mejores pesos. La prueba independiente se evalúa al terminar. Para la explicación matemática de neuronas, gradientes y regularización puedes apoyarte en *Neural Networks from Scratch in Python*, el libro de referencia compartido; este código usa TensorFlow/Keras para llevar esos conceptos a una CNN.

## 3. Dataset: origen, preparación y prueba

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) fue publicado por Zalando Research bajo [licencia MIT](https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE). Contiene 60 000 imágenes originales de entrenamiento y 10 000 de prueba, cada una de 28×28 píxeles y una etiqueta. Los **cuatro archivos IDX comprimidos** se descargan de los [servidores HTTPS que usa Keras](https://github.com/keras-team/keras/blob/master/keras/src/datasets/fashion_mnist.py); si el primero no responde, se intenta el repositorio original. Se contrastan con los [MD5 publicados por Zalando](https://github.com/zalandoresearch/fashion-mnist#readme). Los datos se guardan en `data/fashion-mnist/`, fuera de Git, y solo se vuelven a descargar si faltan o están corruptos. La descarga completa es de alrededor de 30 MB.

**Descarga explícita en Ubuntu, desde la raíz del repositorio** (no requiere Docker, TensorFlow, NumPy ni cuenta de Kaggle):

```bash
python3 -m caso download --data data/fashion-mnist
```

El comando imprime cada archivo y muestra `Fashion-MNIST listo` al terminar. Puedes repetirlo para comprobar los MD5 sin volver a descargar los archivos correctos. `train` consume **solo la copia local**: si falta algo, indica que ejecutes `download`; una vez construida la imagen Docker también es posible entrenar sin Internet. No se incluyen imágenes del dataset en el ZIP del repositorio.

Con la semilla predeterminada 42, el programa baraja **únicamente** las 60 000 imágenes originales de entrenamiento. Usa 50 000 para ajustar los pesos y 10 000 para validación; deja las 10 000 del test oficial separadas hasta el final. Así, la validación guía la parada temprana y el test mide la capacidad de responder a imágenes que no participaron en esa selección. Los píxeles pasan de enteros 0–255 a `float32` en el intervalo [0,1], con forma `(cantidad, 28, 28, 1)`.

Además de la exactitud global se guardan **precisión y recall por categoría**, matriz de confusión y la exactitud de una referencia sencilla: predecir siempre la clase más frecuente del entrenamiento. No ajustes hiperparámetros escogiendo la mejor puntuación de test: usa validación durante los cambios y deja el test para la evaluación final.

## 4. Hardware para entrenar y aplicar el modelo

Equipo previsto: **XENIA XPG 15KC**, Intel Core i7-11800H, 32 GB de RAM, NVIDIA GeForce RTX 3070 con 8 GB de VRAM y **Ubuntu 26.04**. Docker ejecuta una imagen con TensorFlow 2.15 y bibliotecas CUDA compatibles dentro del contenedor; NVIDIA Container Toolkit expone la GPU mediante `--gpus all`. La CPU lee/prepara imágenes y la GPU puede acelerar las operaciones compatibles de la CNN. TensorFlow selecciona una GPU visible automáticamente para operaciones admitidas; el informe registra tanto las GPU detectadas como el dispositivo del tensor de salida de una inferencia.

Comprueba el controlador del host y el acceso desde Docker antes de entrenar:

```bash
nvidia-smi
docker run --rm --gpus all ubuntu:26.04 nvidia-smi
```

La RTX 3070 ya apareció como `/physical_device:GPU:0` en la comprobación anterior de TensorFlow. Eso confirma detección; los tiempos de esta implementación se obtendrán al ejecutar el caso. Para una aplicación de captura en directo también se necesita cámara, iluminación estable, tiempo de carga del modelo y medidas desde la captura hasta el resultado. Una comparación CPU/GPU requiere correr el **mismo** experimento en ambos modos.

## 5. Construcción y entrenamiento en Python

El código reside en [`caso/pipeline.py`](caso/pipeline.py), el [descargador y lector IDX](caso/dataset.py) y la interfaz de consola [`caso/__main__.py`](caso/__main__.py). `download_dataset()` obtiene y verifica los archivos; `load_dataset()` los lee desde disco. `build_model()` crea las capas de la tabla; `train_case()` separa entrenamiento/validación, normaliza y ejecuta `compile`, `fit`, `evaluate` y `predict`. Luego guarda pesos, historial, métricas e informe. Cada ejecución exige una carpeta nueva para no sobrescribir resultados anteriores.

Desde la **raíz de este repositorio**, donde está `Dockerfile`:

```bash
python3 -m caso download --data data/fashion-mnist
docker build -t armando-cnn:tf215 .
mkdir -p .cache outputs
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -e HOME=/app/.cache -v "$PWD:/app" -w /app \
  armando-cnn:tf215 python -m caso device
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -e HOME=/app/.cache -v "$PWD:/app" -w /app \
  armando-cnn:tf215 python -m caso train \
  --data data/fashion-mnist --out outputs/moda-01 \
  --epochs 10 --seed 42 --sample-index 0
```

Para correr en CPU omite `--gpus all` en los dos comandos `docker run`. Si repites la prueba usa otra ruta, por ejemplo `outputs/moda-02`. Los mensajes de TensorFlow sobre registro duplicado de cuDNN, cuFFT o cuBLAS durante el inicio no sustituyen las métricas finales: verifica que el comando termine y genere la carpeta de resultados.

Archivos generados en `outputs/moda-01/`:

| Archivo | Contenido |
| --- | --- |
| `modelo.keras` | Arquitectura y pesos aprendidos |
| `entrada.png`, `salida.json` | Imagen independiente y etiqueta/probabilidades calculadas |
| `metadatos.json` | Clases, semilla, particiones, épocas y dispositivos |
| `metricas.json`, `historial.json` | Exactitud, matriz de confusión, tiempos y curvas por época |
| `informe.md` | Resultados reales y conclusiones correspondientes a esa ejecución |

## 6. Dar una entrada y observar la salida

`train` exporta `entrada.png` del **test oficial** (índice 0 por defecto). También calcula sus diez probabilidades y escribe la etiqueta esperada, la predicha y si acertó en `salida.json`. Para demostrar la aplicación cargando el modelo entrenado **desde disco**, ejecuta:

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -e HOME=/app/.cache -v "$PWD:/app" -w /app \
  armando-cnn:tf215 python -m caso predict \
  --run outputs/moda-01 --image outputs/moda-01/entrada.png
```

El procedimiento es: abrir el PNG, convertirlo a gris, redimensionarlo a 28×28, dividir cada intensidad por 255, agregar las dimensiones de lote y canal, cargar `modelo.keras`, calcular `softmax` y tomar el índice de la mayor probabilidad. Se imprime un JSON con `prediccion` y `probabilidades`. La prueba incluida verifica que guardar y volver a abrir el PNG conserva **exactamente el tensor de entrada** para esta demostración. La salida numérica concreta depende del entrenamiento; revísala en `salida.json` después de ejecutarlo. Para usar otra imagen, cambia `--image`, teniendo presente que fotos comunes se diferencian del dataset de entrenamiento.

### Exportar más imágenes para probar el modelo

Si ya descargaste el dataset con `python3 -m caso download`, puedes sacar ejemplos del **test oficial**, que no se usó para ajustar los pesos. Este comando escoge cinco imágenes distintas de cada una de las diez categorías (50 en total), omite por defecto el índice 0 usado para `entrada.png`, y conserva los píxeles originales en PNG de 28×28:

```bash
mkdir -p outputs
docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -v "$PWD:/app" -w /app \
  armando-cnn:tf215 python -m caso export \
  --data data/fashion-mnist --out outputs/moda-01/imagenes-prueba \
  --per-class 5 --seed 42
```

Abre `outputs/moda-01/imagenes-prueba/catalogo.png` para ver las 50 imágenes ampliadas y sus etiquetas. Cada PNG individual tiene su índice y categoría en el nombre; `etiquetas.csv` guarda las etiquetas verdaderas. El catálogo es **solo una vista previa**: para clasificar, pasa un PNG individual, por ejemplo:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -v "$PWD:/app" -w /app \
  armando-cnn:tf215 python -m caso predict \
  --run outputs/moda-01 \
  --image outputs/moda-01/imagenes-prueba/test_XXXXX_bolsa.png
```

Sustituye `test_XXXXX_bolsa.png` por un nombre real que aparezca en la carpeta. También puedes pasar ese archivo a `scripts/explicar_imagen.py` para que Ollama explique la predicción. Cambia `--per-class` (de 1 a 100) y `--seed` para obtener otras muestras; usa una **ruta nueva en `--out`** en cada extracción, ya que el comando no sobrescribe carpetas. Si tu `entrada.png` se generó con otro `--sample-index`, usa ese número en `--skip-index`. Consulta la etiqueta del CSV solo después de comparar tu predicción, y no escojas hiperparámetros basándote en aciertos repetidos sobre el test.

### Explicar una imagen con la CNN y Ollama

Después de construir la imagen Docker y entrenar al menos una vez, instala y ejecuta [Ollama](https://docs.ollama.com/) en **Ubuntu**. Comprueba que `ollama list` muestre algún modelo local y que su servicio responda (`curl http://127.0.0.1:11434/api/tags`). Desde la raíz del repositorio ejecuta, sustituyendo las rutas y el nombre por los tuyos:

```bash
python3 scripts/explicar_imagen.py /ruta/a/mi-imagen.png --model NOMBRE_DE_OLLAMA
```

La imagen también puede estar fuera del repositorio. Si omites `--model`, el script usa el primer modelo que devuelve Ollama. Para elegir otro entrenamiento o añadir una pregunta al prompt:

```bash
python3 scripts/explicar_imagen.py /ruta/a/mi-imagen.jpg \
  --run outputs/moda-02 --model NOMBRE_DE_OLLAMA \
  --question "¿Qué categoría se predijo y qué tan fiable sería para una foto?"
```

El script [`scripts/explicar_imagen.py`](scripts/explicar_imagen.py) solo necesita Python estándar, Docker y Ollama en el host; **no** instala TensorFlow en Ubuntu. Abre la imagen mediante un volumen Docker de solo lectura, carga `modelo.keras` y `metadatos.json`, y obtiene la categoría y las diez salidas `softmax`. Después envía **solo la categoría, las tres salidas principales y la pregunta** a `/api/generate` de Ollama y muestra la explicación junto con los números originales. Ollama no recibe los píxeles de la imagen. Para estas imágenes pequeñas, la CNN se ejecuta en CPU dentro de Docker, sin `--gpus all`; la GPU del host queda disponible para el LLM y no depende del funcionamiento de NVIDIA Container Toolkit durante la inferencia. Entrenar con GPU sigue usando los comandos de la sección 5.

La red solo aprendió las diez categorías de Fashion-MNIST. Una fotografía, una radiografía o una pieza mecánica se convertirá igualmente en 28×28 en gris y recibirá **alguna** de esas diez etiquetas, aunque no sea una prenda. La respuesta de Ollama interpreta esa salida: no corrige la predicción ni convierte el sistema en un clasificador general o un modelo visual. Para reconocer otros objetos hay que entrenar otra red con datos etiquetados adecuados; para que el LLM inspeccione la imagen en sí se necesitaría un modelo de visión compatible con Ollama y un flujo diferente.

## 7. Conclusiones, velocidad, efectividad y evolución

Lee [`outputs/moda-01/informe.md`](outputs/moda-01/informe.md) **después de entrenar**. Se completa automáticamente con segundos de entrenamiento, exactitud y pérdida de test, comparación con la clase mayoritaria, tiempos de una inferencia (primera, mediana y percentil 95), categorías con menor recall y la entrada/salida del ejemplo. `metricas.json` conserva los valores sin redondear y la matriz de confusión. `historial.json` permite observar sobreajuste: si baja `loss` de entrenamiento mientras `val_loss` empeora, conviene revisar regularización, datos o épocas. Los tiempos de inferencia miden la CNN cargada y la transferencia de su salida a NumPy: **no** incluyen arranque de Docker, lectura del archivo o carga del modelo.

**Mejoras de software:** analizar errores por clase, ampliar datos con transformaciones aplicadas solo al entrenamiento, probar tasa de aprendizaje y arquitectura, y usar transferencia de aprendizaje con imágenes reales. **Mejoras de hardware:** comparar la misma prueba en CPU/GPU, medir consumo y memoria, y mantener el modelo cargado durante la aplicación para evitar recargarlo por cada imagen. Una RTX 3070 no garantiza menor latencia para una imagen diminuta: hay que medirla.

**Escalamiento práctico:** para detectar defectos en piezas impresas en 3D, reúne fotografías propias etiquetadas por tipo de defecto y piezas conformes, controla cámara e iluminación, separa por pieza o lote para que vistas casi idénticas no crucen entrenamiento/prueba, y define el costo de no detectar un defecto. Una foto real necesitará resolución y modelo adecuados, nuevos datos, validación independiente y medición del proceso completo. La red de ropa de este repositorio no reconoce defectos sin ese trabajo.

## Reproducibilidad y límites de verificación

Puedes ejecutar las comprobaciones locales sin TensorFlow con `python3 -m unittest discover -s tests -v`. La descarga requiere acceso a los servidores de Fashion-MNIST desde Ubuntu solo una vez; el entrenamiento real, aceleración de GPU y cifras de exactitud/latencia requieren ejecutar el comando Docker en la XENIA. Ninguna medición de esa laptop se inventa en el repositorio. El código es una implementación nueva de este caso; no incluye ni redistribuye el libro de referencia ni las imágenes de Fashion-MNIST.
