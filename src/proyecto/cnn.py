"""CNN TensorFlow basada en examples/red-neuronal-convolucional/caso/pipeline.py."""
import io
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from proyecto.training import load_run

CATALOG = {
    'fashion_mnist': {
        'name': 'Fashion-MNIST · prendas',
        'description': '60 000 imágenes de entrenamiento y 10 000 de prueba, 28×28 en gris. Descarga aproximada: 30 MB.',
        'source': 'https://github.com/zalandoresearch/fashion-mnist',
        'classes': ['camiseta', 'pantalón', 'suéter', 'vestido', 'abrigo', 'sandalia', 'camisa', 'tenis', 'bolsa', 'botín'],
    },
    'mnist': {
        'name': 'MNIST · dígitos manuscritos',
        'description': '60 000 imágenes de entrenamiento y 10 000 de prueba, 28×28 en gris. Descarga aproximada: 11 MB.',
        'source': 'https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data',
        'classes': [str(i) for i in range(10)],
    },
}


def data_root():
    return Path(os.environ.get('NN2GUI_DATA', 'data'))


def tensorflow():
    # Limita los hilos tanto en operaciones como en los datasets de entrenamiento.
    os.environ.setdefault('KERAS_HOME', str((data_root() / 'datasets' / 'keras').resolve()))
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
    import tensorflow as tf
    return tf


def load_dataset(dataset):
    if dataset not in CATALOG:
        raise ValueError('Dataset no compatible con esta CNN.')
    if dataset == 'fashion_mnist':
        from proyecto.fashion_dataset import download_dataset, load_dataset as load_local
        directory = data_root() / 'datasets' / dataset
        download_dataset(directory)
        return load_local(directory)
    tf = tensorflow()
    directory = data_root() / 'datasets'
    directory.mkdir(parents=True, exist_ok=True)
    return tf.keras.datasets.mnist.load_data(path='mnist.npz')


def prepare_pixels(pixels):
    array = np.asarray(pixels)
    if array.shape != (28, 28) or not np.isfinite(array).all():
        raise ValueError('Se requiere una imagen de 28×28 con píxeles finitos.')
    if array.min() < 0 or array.max() > 255:
        raise ValueError('Los píxeles deben estar entre 0 y 255.')
    return array.astype(np.float32)[None, ..., None] / 255.0


def prepare_image(content):
    with Image.open(io.BytesIO(content)) as image:
        return prepare_pixels(image.convert('L').resize((28, 28), Image.Resampling.BILINEAR))


def build_model(filters, dense, dropout, rate):
    if not 1 <= len(filters) <= 3 or any(n not in (8, 16, 32, 64, 128) for n in filters):
        raise ValueError('Usa de 1 a 3 bloques con filtros admitidos.')
    if not 8 <= dense <= 256 or not 0 <= dropout <= 0.7 or not 0 < rate <= 0.1:
        raise ValueError('Configuración de CNN inválida.')
    tf = tensorflow()
    layers = [tf.keras.Input(shape=(28, 28, 1))]
    # Dos bloques reproducen el ejemplo; tres necesitan padding para caber en 28×28.
    for count in filters:
        layers += [tf.keras.layers.Conv2D(count, 3, padding='same' if len(filters) == 3 else 'valid', activation='relu'),
                   tf.keras.layers.MaxPooling2D()]
    layers += [tf.keras.layers.Flatten(), tf.keras.layers.Dense(dense, activation='relu'),
               tf.keras.layers.Dropout(dropout), tf.keras.layers.Dense(10, activation='softmax')]
    model = tf.keras.Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_cnn(dataset, filters=(32, 64), dense=64, dropout=0.2, epochs=5,
              batch_size=128, rate=0.001, seed=42, sample_count=5000, progress=None):
    if dataset not in CATALOG or not 1 <= epochs <= 100 or batch_size not in (16, 32, 64, 128):
        raise ValueError('Dataset o parámetros de entrenamiento no admitidos.')
    if not 200 <= sample_count <= 60000:
        raise ValueError('Selecciona de 200 a 60 000 imágenes de entrenamiento.')
    tf = tensorflow()
    tf.keras.utils.set_random_seed(seed)
    model = build_model(filters, dense, dropout, rate)
    (images, labels), (test_images, test_labels) = load_dataset(dataset)
    indices = np.arange(len(images))
    if sample_count < len(indices):
        indices, _ = train_test_split(indices, train_size=sample_count, stratify=labels, random_state=seed)
    fit, val = train_test_split(indices, test_size=1/6, stratify=labels[indices], random_state=seed)
    def batches(raw, targets, shuffle=False):
        # Normalizar por lote evita mantener copias float32 del dataset completo.
        ds = tf.data.Dataset.from_tensor_slices((raw, targets))
        if shuffle:
            ds = ds.shuffle(min(len(raw), 5000), seed=seed)
        ds = ds.batch(batch_size).map(lambda x, y: (tf.cast(x[..., None], tf.float32) / 255.0, y))
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        return ds.with_options(options)
    history = []
    class Progress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            if not np.isfinite([logs['loss'], logs['val_loss']]).all():
                raise ValueError('El entrenamiento divergió; reduce la tasa de aprendizaje.')
            history.append({'época': epoch + 1, 'entrenamiento': float(logs['loss']), 'validación': float(logs['val_loss'])})
            if progress:
                progress(history)
    model.fit(batches(images[fit], labels[fit], True), validation_data=batches(images[val], labels[val]),
              epochs=epochs, verbose=0, shuffle=False, callbacks=[Progress(), tf.keras.callbacks.EarlyStopping(
                  monitor='val_loss', patience=3, restore_best_weights=True)])
    predictions = []
    for x, _ in batches(test_images, test_labels):
        predictions.extend(model(x, training=False).numpy().argmax(1).tolist())
    metrics = {'accuracy': float(np.mean(np.asarray(predictions) == test_labels)),
               'confusion': confusion_matrix(test_labels, predictions, labels=list(range(10))).tolist()}
    config = {'kind': 'cnn', 'framework': 'tensorflow', 'dataset': dataset,
              'source': CATALOG[dataset]['source'], 'classes': CATALOG[dataset]['classes'],
              'input_shape': [28, 28, 1], 'preprocessing': 'grayscale_resize_bilinear_div255',
              'filters': list(filters), 'dense': dense, 'dropout': dropout, 'rate': rate,
              'epochs': epochs, 'epochs_ran': len(history), 'batch_size': batch_size, 'seed': seed,
              'sample_count': len(indices), 'split': {'train': fit.tolist(), 'validation': val.tolist(),
              'test': list(range(len(test_images)))}, 'test_partition': 'official_test',
              'versions': {'tensorflow': tf.__version__, 'numpy': np.__version__}, 'device': 'CPU'}
    directory = data_root() / uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    model.save(directory / 'model.keras')
    Image.fromarray(test_images[0]).save(directory / 'sample.png')
    config['sample_expected'] = config['classes'][int(test_labels[0])]
    for name, value in [('config', config), ('metrics', metrics), ('history', history)]:
        (directory / f'{name}.json').write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tf.keras.backend.clear_session()
    return directory.name


def predict_cnn(run_id, content):
    config, _, _ = load_run(run_id)
    if config.get('kind') != 'cnn':
        raise ValueError('Esta ejecución no es una CNN.')
    tf = tensorflow()
    model = tf.keras.models.load_model(data_root() / run_id / 'model.keras', compile=False, safe_mode=True)
    return model(prepare_image(content), training=False).numpy()[0]
