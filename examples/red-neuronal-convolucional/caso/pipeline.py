"""Dataset, CNN, evaluación y predicción del caso práctico."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image


CLASSES = ("camiseta", "pantalón", "suéter", "vestido", "abrigo",
           "sandalia", "camisa", "tenis", "bolsa", "botín")


def prepare_pixels(pixels):
    """De una imagen L de 28×28 a un lote de float32 en [0,1]."""
    array = np.asarray(pixels)
    if array.shape != (28, 28):
        raise ValueError(f"Se requiere una imagen de 28×28; recibida: {array.shape}")
    return array.astype(np.float32)[None, ..., None] / 255.0


def prepare_file(path):
    """Usa el mismo escalado que el entrenamiento para una imagen externa."""
    with Image.open(path) as source:
        image = source.convert("L").resize((28, 28), Image.Resampling.BILINEAR)
        return prepare_pixels(image)


def build_model(tf):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def report_text(metadata, metrics, sample, history):
    """Completa las observaciones con los resultados de ESTA ejecución."""
    weakest = sorted(CLASSES, key=lambda name: metrics["per_class"][name]["recall"])[:3]
    table = "\n".join(
        f"| {name} | {metrics['per_class'][name]['precision']:.3f} | "
        f"{metrics['per_class'][name]['recall']:.3f} | "
        f"{int(metrics['per_class'][name]['support'])} |" for name in CLASSES
    )
    top = sorted(sample["probabilities"].items(), key=lambda item: -item[1])[:3]
    top_text = ", ".join(f"{name}: {value:.1%}" for name, value in top)
    gpu = ", ".join(metadata["gpus"]) or "ninguna; ejecución en CPU"
    delta = 100 * (metrics["test_accuracy"] - metrics["baseline_accuracy"])
    return f"""# Resultados del caso práctico de Armando

La explicación de los siete requisitos está en [README.md](../../README.md). Esta página se generó a partir de la ejecución real; no contiene cifras previstas.

## Configuración

- Dataset abierto: Fashion-MNIST de Zalando Research ({metadata['dataset_source']}); archivos locales en `{metadata['dataset_path']}`.
- Semilla: {metadata['seed']}; épocas realizadas: {metadata['epochs_ran']} de {metadata['epochs_limit']} posibles.
- Partición: {metadata['train_count']:,} entrenamiento, {metadata['validation_count']:,} validación, {metadata['test_count']:,} prueba.
- GPU detectada: {gpu}. Tensor de salida de la inferencia en: `{metadata['output_device']}`.

## Velocidad y efectividad

| Medida | Resultado |
| --- | ---: |
| Entrenamiento (`model.fit`) | {metrics['train_seconds']:.2f} s |
| Evaluación de 10 000 imágenes | {metrics['evaluation_seconds']:.2f} s |
| Exactitud del test | {metrics['test_accuracy']:.2%} |
| Pérdida del test | {metrics['test_loss']:.4f} |
| Referencia: clase mayoritaria «{metrics['baseline_class']}» | {metrics['baseline_accuracy']:.2%} |
| Primera inferencia, modelo cargado | {metrics['cold_inference_ms']:.2f} ms |
| Mediana de 30 inferencias individuales | {metrics['median_inference_ms']:.2f} ms |
| Percentil 95 de 30 inferencias | {metrics['p95_inference_ms']:.2f} ms |

La exactitud supera la referencia en {delta:.2f} puntos porcentuales. Las latencias incluyen la ejecución del modelo y la transferencia del resultado a NumPy; excluyen la apertura del archivo, la carga del modelo desde disco y el inicio de Docker. La mediana y el percentil 95 proceden de una sola imagen, cinco pasadas de calentamiento y 30 repeticiones.

| Clase | Precisión | Recall | Imágenes del test |
| --- | ---: | ---: | ---: |
{table}

En esta ejecución, las clases con menor recall son: {', '.join(weakest)}. Revisa `metricas.json` para ver la matriz de confusión completa y `historial.json` para comparar `loss` con `val_loss` durante las {len(history['loss'])} épocas. Ajusta arquitectura o regularización con validación; reserva el test para la evaluación final.

## Imagen de entrada y predicción

La [imagen original de prueba](entrada.png) corresponde al índice {sample['index']} del test oficial. Clase esperada: **{sample['expected']}**. Predicción: **{sample['predicted']}** ({'acierto' if sample['correct'] else 'error'}). Probabilidades más altas: {top_text}. Las diez salidas están en `salida.json`.

El procedimiento fue: leer 28×28 píxeles en gris, dividir cada valor por 255, agregar las dimensiones de lote y canal, ejecutar la CNN, y escoger el índice con mayor salida `softmax`. Para comprobar la predicción desde el PNG guardado usa `python -m caso predict --run <carpeta> --image <carpeta>/entrada.png`.

## Conclusiones y mejoras

Estos resultados describen imágenes del mismo origen que Fashion-MNIST. Para fotografías reales, recoge nuevas imágenes etiquetadas y mide de nuevo efectividad y latencia completa, desde cámara hasta salida. En software: compara aumento de datos solo en entrenamiento, tasa de aprendizaje, pesos preentrenados y clases prioritarias. En hardware: mide la misma tarea en CPU y GPU, considera consumo, memoria y tiempo de captura. Para inspección de piezas impresas en 3D, añade cámara e iluminación fija, define tipos de defecto y divide por pieza o lote para evitar que vistas casi idénticas se repartan entre entrenamiento y prueba.
"""


def train_case(out, epochs=10, seed=42, sample_index=0, data="data/fashion-mnist"):
    """Entrena una CNN nueva, prueba con datos independientes y guarda la evidencia."""
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix

    from .dataset import load_dataset

    if epochs < 1 or not 0 <= sample_index < 10000:
        raise ValueError("Se necesitan épocas positivas y --sample-index entre 0 y 9999.")
    dest = Path(out)
    if dest.exists():
        raise FileExistsError(f"La carpeta {dest} ya existe; usa otro nombre de ejecución.")

    tf.keras.utils.set_random_seed(seed)
    (train_raw, train_labels), (test_raw, test_labels) = load_dataset(data)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_raw))
    fit_idx, val_idx = order[:50000], order[50000:]
    x_fit = train_raw[fit_idx].astype(np.float32)[..., None] / 255.0
    x_val = train_raw[val_idx].astype(np.float32)[..., None] / 255.0
    x_test = test_raw.astype(np.float32)[..., None] / 255.0

    model = build_model(tf)
    started = perf_counter()
    history = model.fit(x_fit, train_labels[fit_idx], validation_data=(x_val, train_labels[val_idx]),
                        batch_size=128, epochs=epochs, verbose=2,
                        callbacks=[tf.keras.callbacks.EarlyStopping(
                            monitor="val_loss", patience=3, restore_best_weights=True)])
    fit_seconds = perf_counter() - started

    started = perf_counter()
    loss, accuracy = model.evaluate(x_test, test_labels, verbose=0)
    evaluation_seconds = perf_counter() - started
    predicted = model.predict(x_test, batch_size=256, verbose=0).argmax(axis=1)
    majority_idx = int(np.bincount(train_labels[fit_idx], minlength=10).argmax())

    tensor = tf.convert_to_tensor(x_test[sample_index:sample_index + 1])
    started = perf_counter()
    output = model(tensor, training=False)
    probabilities = output.numpy()[0]
    cold_ms = 1000 * (perf_counter() - started)
    for _ in range(5):
        model(tensor, training=False).numpy()
    timings = []
    for _ in range(30):
        started = perf_counter()
        output = model(tensor, training=False)
        probabilities = output.numpy()[0]
        timings.append(1000 * (perf_counter() - started))

    metrics = {
        "test_loss": float(loss), "test_accuracy": float(accuracy),
        "train_seconds": fit_seconds, "evaluation_seconds": evaluation_seconds,
        "baseline_class": CLASSES[majority_idx],
        "baseline_accuracy": float(np.mean(test_labels == majority_idx)),
        "cold_inference_ms": cold_ms, "median_inference_ms": float(np.median(timings)),
        "p95_inference_ms": float(np.percentile(timings, 95)),
        "per_class": classification_report(test_labels, predicted, labels=list(range(10)),
                                           target_names=CLASSES, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(test_labels, predicted, labels=list(range(10))).tolist(),
    }
    meta = {"dataset": "Fashion-MNIST", "dataset_path": str(Path(data)),
            "dataset_source": "https://github.com/zalandoresearch/fashion-mnist",
            "classes": CLASSES, "seed": seed,
            "train_count": len(fit_idx), "validation_count": len(val_idx), "test_count": len(test_raw),
            "epochs_limit": epochs, "epochs_ran": len(history.history["loss"]),
            "gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
            "output_device": output.device}
    sample = {"index": sample_index, "expected": CLASSES[int(test_labels[sample_index])],
              "predicted": CLASSES[int(np.argmax(probabilities))],
              "correct": bool(np.argmax(probabilities) == test_labels[sample_index]),
              "probabilities": dict(zip(CLASSES, map(float, probabilities)))}

    dest.mkdir(parents=True)
    model.save(dest / "modelo.keras")
    Image.fromarray(test_raw[sample_index]).save(dest / "entrada.png")
    for filename, data in (("metadatos.json", meta), ("metricas.json", metrics),
                           ("historial.json", history.history), ("salida.json", sample)):
        (dest / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    (dest / "informe.md").write_text(report_text(meta, metrics, sample, history.history), encoding="utf-8")
    print(json.dumps({"resultados": str(dest), "exactitud": metrics["test_accuracy"],
                      "prediccion_ejemplo": sample["predicted"], "gpu": meta["gpus"]},
                     indent=2, ensure_ascii=False))


def predict_file(run, image):
    """Carga los pesos guardados y devuelve la etiqueta y diez probabilidades."""
    import tensorflow as tf

    run_path = Path(run)
    metadata = json.loads((run_path / "metadatos.json").read_text(encoding="utf-8"))
    array = prepare_file(image)
    model = tf.keras.models.load_model(run_path / "modelo.keras")
    probabilities = model.predict(array, verbose=0)[0]
    classes = metadata["classes"]
    return {"imagen": str(image), "prediccion": classes[int(np.argmax(probabilities))],
            "probabilidades": dict(zip(classes, map(float, probabilities)))}
