# Guía práctica y fundamento del clasificador de prendas

**Proyecto:** Armando — Implementación de un caso práctico de red neuronal.  
**Repositorio:** ejecutar todos los comandos desde la carpeta que contiene `Dockerfile`, `README.md` y `caso/`.  
**Alcance:** clasificar imágenes similares a Fashion-MNIST en diez categorías de prendas y, de manera opcional, explicar la predicción mediante Ollama. El modelo **no** reconoce cualquier imagen ni aprende un dataset distinto sin adaptar el código y volverlo a entrenar.

Esta guía complementa el [README](README.md). Los apartados 1 a 7 responden directamente al ejercicio; al final hay procedimientos reproducibles para ejecutar, repetir y evaluar el caso.

## 1. Caso práctico: qué problema resuelve y por qué elegirlo

La tarea es **clasificación de imágenes**: dada una imagen pequeña en escala de grises, la red asigna una de las clases camiseta, pantalón, suéter, vestido, abrigo, sandalia, camisa, tenis, bolsa o botín. Elegí Fashion-MNIST porque se obtiene sin crear una cuenta, tiene etiquetas para comprobar las predicciones y permite estudiar un ciclo completo de aprendizaje visual en un equipo personal. La entrada esperada es de **28 × 28 × 1** píxeles; la salida consiste en diez valores `softmax` y la clase con el valor más alto.

La utilidad didáctica consiste en poder seguir una misma imagen desde su etiqueta original hasta la decisión de la CNN, medir errores por categoría y comprobar qué cambia al volver a entrenar. El ejemplo no demuestra rendimiento en fotografías reales: su distribución, resolución, fondo e iluminación pueden ser muy diferentes.

## 2. Tipo de red y de entrenamiento

Se usa una **red neuronal convolucional (CNN)** implementada con TensorFlow/Keras en [`caso/pipeline.py`](caso/pipeline.py). El aprendizaje es **supervisado**, pues durante el ajuste cada imagen tiene una etiqueta conocida de 0 a 9. La CNN aprende filtros locales en capas convolucionales; las capas de agrupación reducen el tamaño espacial y las densas combinan los rasgos. No hay entrenamiento de un LLM: Ollama solo redacta después una interpretación de la predicción.

| Etapa | Forma aproximada de salida | Función |
| --- | --- | --- |
| Entrada | 28 × 28 × 1 | Imagen gris normalizada entre 0 y 1 |
| `Conv2D(32, 3, relu)` y `MaxPooling2D` | 13 × 13 × 32 | Bordes y patrones locales |
| `Conv2D(64, 3, relu)` y `MaxPooling2D` | 5 × 5 × 64 | Rasgos más complejos |
| `Flatten`, `Dense(64, relu)`, `Dropout(0.2)` | 64 | Combinar rasgos y regularizar |
| `Dense(10, softmax)` | 10 valores | Puntuar las diez clases |

El optimizador **Adam** modifica los pesos por retropropagación para reducir `sparse_categorical_crossentropy`. El entrenamiento usa lotes de 128 muestras y, de forma predeterminada, un máximo de diez épocas. `EarlyStopping` detiene el ajuste si la pérdida de validación no mejora durante tres épocas y recupera los mejores pesos. `accuracy` mide el porcentaje de clasificaciones correctas. Las salidas `softmax` permiten ordenar clases, pero no son garantías ni una medida de confianza calibrada.

## 3. Obtención y separación de los datos

El conjunto abierto [Fashion-MNIST de Zalando Research](https://github.com/zalandoresearch/fashion-mnist) aporta 60 000 imágenes de entrenamiento y 10 000 de prueba. El módulo [`caso/dataset.py`](caso/dataset.py) descarga cuatro archivos IDX comprimidos desde las fuentes configuradas, comprueba su MD5 antes de usarlos y conserva la copia en `data/fashion-mnist/`. El ZIP del repositorio **no contiene** el dataset; la descarga se hace una sola vez, salvo que falte o esté corrupto algún archivo.

```bash
python3 -m caso download --data data/fashion-mnist
```

De las 60 000 imágenes originales de entrenamiento, el programa baraja con una semilla y destina **50 000 a ajustar los pesos y 10 000 a validación**. Las otras **10 000 imágenes del test oficial** se reservan para la evaluación final y para los ejemplos exportados. Los valores enteros 0–255 se convierten a `float32` y se dividen por 255. La separación evita que la red aprenda directamente las imágenes usadas para juzgar su resultado final. Cambios de arquitectura y épocas deben decidirse con validación, sin escoger la versión ganadora a partir del test.

## 4. Hardware y sistema de ejecución

El equipo de referencia es la **XENIA XPG 15KC**, con Intel Core i7-11800H, 32 GB de RAM, NVIDIA RTX 3070 de 8 GB y Ubuntu 26.04. La CPU lee y prepara datos; una GPU NVIDIA accesible desde Docker puede acelerar operaciones de entrenamiento. El `Dockerfile` usa `tensorflow/tensorflow:2.15.0-gpu` e instala las versiones de NumPy, Pillow y scikit-learn fijadas en `requirements.txt`. No es necesario instalar TensorFlow en Ubuntu.

Comprueba el acceso a la GPU en el host y desde Docker:

```bash
nvidia-smi
docker run --rm --gpus all ubuntu:26.04 nvidia-smi
```

Si el host indica *Driver/library version mismatch* o Docker no consigue iniciar un contenedor con `--gpus all`, corrige primero la instalación NVIDIA si quieres usar la GPU. Para **seguir trabajando en CPU**, omite `--gpus all` del comando de entrenamiento; no necesitas GPU para descargar ejemplos, predecir una imagen pequeña o ejecutar el script que consulta a Ollama. La inferencia del script de Ollama corre en CPU y deja libre la GPU para el modelo de lenguaje. La detección de una GPU por TensorFlow no implica por sí sola una mejora medible de velocidad.

## 5. Construcción en Python y entrenamiento

La interfaz de consola está en [`caso/__main__.py`](caso/__main__.py), el lector del dataset en [`caso/dataset.py`](caso/dataset.py) y la CNN en [`caso/pipeline.py`](caso/pipeline.py). `build_model()` define y compila la red; `train_case()` fija la semilla, separa los datos, normaliza, llama a `fit()`, evalúa con `evaluate()` y guarda el modelo y los resultados. Todos los comandos siguientes parten de la raíz del repositorio.

**Primera preparación:**

```bash
python3 -m caso download --data data/fashion-mnist
docker build -t armando-cnn:tf215 .
mkdir -p .cache outputs
```

**Primer entrenamiento con GPU accesible:**

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -e HOME=/app/.cache -v "$PWD:/app" -w /app \
  armando-cnn:tf215 python -m caso train \
  --data data/fashion-mnist --out outputs/moda-01 \
  --epochs 10 --seed 42 --sample-index 0
```

Para entrenar en **CPU**, usa el mismo comando quitando `--gpus all`. La carpeta elegida en `--out` debe ser nueva. Se crean `modelo.keras` (arquitectura y pesos), `metadatos.json`, `metricas.json`, `historial.json`, `informe.md`, `entrada.png` y `salida.json` dentro de `outputs/moda-01/`. Los errores de registro duplicado de cuDNN/cuFFT/cuBLAS al iniciar TensorFlow pueden aparecer aunque el comando finalice; comprueba el código de salida y los archivos generados.

### Volver a entrenar y comparar ejecuciones

En este repositorio **volver a entrenar significa iniciar una CNN nueva desde pesos iniciales**, conservando intacta la ejecución anterior. No hay un comando para reanudar `modelo.keras` ni para continuar ajustando sus pesos. Por ejemplo, para probar un máximo de 15 épocas con la misma semilla y la misma partición:

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -e HOME=/app/.cache -v "$PWD:/app" -w /app \
  armando-cnn:tf215 python -m caso train \
  --data data/fashion-mnist --out outputs/moda-02 \
  --epochs 15 --seed 42 --sample-index 0
```

Quita `--gpus all` si entrenas en CPU. Al comparar `moda-01` y `moda-02`, revisa `metricas.json`, `historial.json` e `informe.md` de **ambas** carpetas. Cambiar una variable a la vez facilita atribuir diferencias; incluso con la misma semilla, GPU y CPU o distintas bibliotecas pueden introducir variación numérica. `EarlyStopping` puede detenerse antes del máximo de épocas. No repitas pruebas hasta elegir por su exactitud en el test: reserva el test para describir el resultado final.

**Otro dataset:** cambiar `--data` no hace que el programa acepte datos arbitrarios; espera los cuatro archivos IDX y las diez clases de Fashion-MNIST. Para usar un conjunto distinto debes adaptar `load_dataset()`, las clases y la capa final, el preprocesamiento y las métricas; después vuelve a entrenar y evalúa con una separación propia de ese conjunto.

## 6. Entrada, salida y uso del modelo entrenado

La ejecución `train` exporta `outputs/moda-01/entrada.png`, una imagen del test oficial elegida por `--sample-index`. `salida.json` guarda su etiqueta real, la predicción y los diez valores generados. Para **cargar desde disco** la CNN ya entrenada y volver a clasificar ese archivo:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -v "$PWD:/app" -w /app \
  armando-cnn:tf215 python -m caso predict \
  --run outputs/moda-01 --image outputs/moda-01/entrada.png
```

`predict_file()` abre la imagen con Pillow, la convierte a gris, la redimensiona a 28 × 28, normaliza los píxeles, añade lote y canal, carga `modelo.keras` y calcula las diez salidas de `softmax`. La clase presentada es la posición del valor máximo. El JSON impreso contiene `imagen`, `prediccion` y `probabilidades`; compara esa salida con `salida.json`. Si usas otra ejecución, cambia `--run` por su carpeta. Una imagen que no sea ropa del estilo de Fashion-MNIST también recibirá **alguna** de las diez etiquetas, aunque esa respuesta no sea útil.

### Exportar más imágenes del dataset para probar

El comando `export` elige, con semilla reproducible, cinco PNG del test **por categoría**, para un total de 50. Omite el índice 0 por defecto para no repetir la `entrada.png` predeterminada. `catalogo.png` es una vista ampliada con etiquetas; **no la pases al clasificador como si fuera una prenda individual**. Los PNG sueltos conservan el tamaño y los píxeles originales; `etiquetas.csv` registra el índice y la clase verdadera de cada uno.

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -v "$PWD:/app" -w /app \
  armando-cnn:tf215 python -m caso export \
  --data data/fashion-mnist --out outputs/moda-01/imagenes-prueba \
  --per-class 5 --seed 42
```

Abre `outputs/moda-01/imagenes-prueba/catalogo.png` y consulta los **nombres reales** con:

```bash
ls outputs/moda-01/imagenes-prueba/test_*.png
```

Después clasifica uno de los archivos listados, sustituyendo `test_XXXXX_bolsa.png` por su nombre exacto:

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -v "$PWD:/app" -w /app \
  armando-cnn:tf215 python -m caso predict \
  --run outputs/moda-01 \
  --image outputs/moda-01/imagenes-prueba/test_XXXXX_bolsa.png
```

`--per-class` acepta entre 1 y 100. Para obtener otra muestra, cambia `--seed` y elige una **ruta nueva** con `--out` (el comando no sobrescribe la anterior). Si `entrada.png` se creó con otro `--sample-index`, pásalo mediante `--skip-index`. El catálogo y el CSV revelan las respuestas correctas: úsalos para comprobar ejemplos, sin convertir este test en datos de entrenamiento.

### Obtener una explicación en español mediante Ollama

Se requiere Ollama en **Ubuntu**, un modelo de lenguaje local descargado, Docker y `modelo.keras` ya entrenado. Comprueba el servicio y copia el nombre exacto de un modelo de `ollama list`:

```bash
ollama list
curl http://127.0.0.1:11434/api/tags
```

Desde la raíz del repositorio, pasa un PNG individual, tanto del dataset exportado como una ruta absoluta externa:

```bash
python3 scripts/explicar_imagen.py \
  outputs/moda-01/imagenes-prueba/test_XXXXX_bolsa.png \
  --model NOMBRE_DE_OLLAMA
```

Para plantear una pregunta concreta y usar otro entrenamiento:

```bash
python3 scripts/explicar_imagen.py /ruta/a/imagen.jpg \
  --run outputs/moda-02 --model NOMBRE_DE_OLLAMA \
  --question "¿Cuál fue la predicción y qué límites tiene para esta foto?"
```

La ruta de imagen puede estar fuera del repositorio, pero la carpeta `--run` debe estar **dentro** del repositorio. Si omites `--model`, se elige el primer modelo que informa Ollama. El script primero llama a `python -m caso predict` en Docker **sin `--gpus all`** y luego envía a Ollama un prompt con la categoría predicha, las tres salidas `softmax` principales y tu pregunta. Imprime la predicción numérica y la respuesta del LLM. **Ollama no recibe los píxeles** y no valida por sí mismo la clasificación: redacta a partir de esos datos. Un LLM visual requeriría un flujo diferente que envíe efectivamente la imagen a un modelo con capacidad de visión.

## 7. Conclusiones, velocidad, efectividad y posibles mejoras

**Efectividad.** El archivo `metricas.json` contiene `test_accuracy`, `test_loss`, precisión y recall por clase, matriz de confusión y una referencia que siempre predice la clase más frecuente. `historial.json` permite comparar pérdidas de entrenamiento y validación. Si una clase tiene bajo recall, estudia sus errores concretos. Los números de un entrenamiento **deben tomarse de tu propia ejecución**; la guía no presupone una exactitud.

**Velocidad.** En `metricas.json` se guardan `train_seconds`, `evaluation_seconds`, `cold_inference_ms`, `median_inference_ms` y `p95_inference_ms`. Las medidas de inferencia se toman con la CNN ya cargada; **no incluyen** el inicio de Docker, apertura de archivo ni carga del modelo, y tampoco la generación de texto por Ollama. Para medir el tiempo real que experimenta una persona, cronometra también el comando completo que usa el LLM, con el modelo ya descargado y tanto en frío como en caliente. La velocidad del LLM depende de su modelo y de la memoria disponible; no hay cifras medidas de tu laptop incluidas aquí.

| Qué registrar en tu ejecución | Dónde encontrarlo | Valor |
| --- | --- | --- |
| Exactitud y pérdida del test | `outputs/moda-01/metricas.json` | Completar tras entrenar |
| Clase con menor recall | `outputs/moda-01/informe.md` | Completar tras entrenar |
| Segundos de entrenamiento | `outputs/moda-01/metricas.json` | Completar tras entrenar |
| Mediana y percentil 95 de inferencia CNN | `outputs/moda-01/metricas.json` | Completar tras entrenar |
| Tiempo total con Ollama | Medición externa del comando completo | Completar al probarlo |

**Mejoras de software:** analizar la matriz de confusión, añadir aumentos de datos solo al entrenamiento, estudiar la tasa de aprendizaje, calibrar las salidas si se necesita incertidumbre útil y mantener la CNN cargada para muchas consultas. Para el LLM, limitar su respuesta a la información de la CNN y medir por separado el tiempo de generación. **Mejoras de hardware:** comparar la misma ejecución en CPU y GPU, vigilar VRAM y memoria, y escoger un LLM local que quepa en la GPU. Para una sola imagen 28 × 28, el costo de iniciar Docker y cargar modelos puede dominar al cálculo de la CNN.

**Escalar a un problema real:** por ejemplo, detectar defectos en piezas impresas en 3D. Reunir fotografías propias etiquetadas por clase, definir pieza sana y defectos relevantes, controlar iluminación, separar por pieza o lote para evitar fuga de información, aumentar la resolución y adaptar el modelo o usar aprendizaje por transferencia. Después medir precisión, recall de defectos, errores de alto costo y latencia desde la cámara hasta la respuesta. La CNN actual de ropa **no** detectará defectos sin esos datos y ese entrenamiento.

## Comprobaciones y referencias dentro del repositorio

Ejecuta `python3 -m unittest discover -s tests -v` para las pruebas locales del descargador, preprocesamiento, exportación y puente con Ollama. La evaluación cuantitativa y los tiempos reales requieren ejecutar `train` y consultar el informe generado en tu equipo. Para ampliar la base teórica de neuronas, función de pérdida y retropropagación, consulta el libro de referencia que compartiste, *Neural Networks from Scratch in Python*; esta implementación utiliza TensorFlow/Keras en vez de reproducir su código.
