"""Descarga verificable de Fashion-MNIST y lectura local del formato IDX."""

import gzip
import hashlib
import os
import struct
import tempfile
from pathlib import Path
from urllib.request import urlopen


# Publicados por Zalando Research junto con los cuatro archivos del dataset.
FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
BASE_URLS = (
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/",
    "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
)
MAX_COMPRESSED_BYTES = 50 * 1024 * 1024


def md5_file(path):
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_one(directory, filename, checksum, bases=BASE_URLS):
    """Descarga a un archivo temporal; reemplaza el destino solo si coincide MD5."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    if target.is_file() and md5_file(target) == checksum:
        print(f"Ya descargado y verificado: {target}", flush=True)
        return target

    errors = []
    for base in bases:
        temp_name = None
        url = base + filename
        try:
            print(f"Descargando {url}", flush=True)
            digest = hashlib.md5(usedforsecurity=False)
            with urlopen(url, timeout=60) as response, tempfile.NamedTemporaryFile(
                mode="wb", dir=directory, prefix=filename + ".", suffix=".part", delete=False
            ) as temp:
                temp_name = temp.name
                received = 0
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    received += len(chunk)
                    if received > MAX_COMPRESSED_BYTES:
                        raise ValueError("Descarga demasiado grande para Fashion-MNIST")
                    digest.update(chunk)
                    temp.write(chunk)
            if digest.hexdigest() != checksum:
                raise ValueError(f"MD5 distinto al publicado: {digest.hexdigest()}")
            os.replace(temp_name, target)
            print(f"Verificado: {target} ({received:,} bytes)", flush=True)
            return target
        except (OSError, ValueError) as exc:
            errors.append(f"{url}: {exc}")
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)
    raise RuntimeError("No se pudo obtener " + filename + ".\n" + "\n".join(errors))


def download_dataset(directory):
    """Descarga una vez, comprueba MD5 y reutiliza los archivos correctos."""
    paths = {name: download_one(directory, name, checksum)
             for name, checksum in FILES.items()}
    print(f"Fashion-MNIST listo en {Path(directory)}", flush=True)
    return paths


def read_images(path):
    """Lee un archivo IDX-3; valida cabecera y longitud antes de darle forma."""
    import numpy as np

    with gzip.open(path, "rb") as stream:
        header = stream.read(16)
        if len(header) != 16:
            raise ValueError(f"Cabecera IDX incompleta: {path}")
        magic, count, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051 or (rows, cols) != (28, 28):
            raise ValueError(f"Imagen IDX inesperada: {path}")
        pixels = stream.read()
    if len(pixels) != count * rows * cols:
        raise ValueError(f"Número de píxeles incorrecto: {path}")
    return np.frombuffer(pixels, dtype=np.uint8).reshape(count, rows, cols)


def read_labels(path):
    """Lee un archivo IDX-1 y verifica sus etiquetas."""
    import numpy as np

    with gzip.open(path, "rb") as stream:
        header = stream.read(8)
        if len(header) != 8:
            raise ValueError(f"Cabecera IDX incompleta: {path}")
        magic, count = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"Etiquetas IDX inesperadas: {path}")
        labels = stream.read()
    if len(labels) != count:
        raise ValueError(f"Número de etiquetas incorrecto: {path}")
    values = np.frombuffer(labels, dtype=np.uint8)
    if np.any(values > 9):
        raise ValueError(f"Etiqueta fuera de 0–9: {path}")
    return values


def load_dataset(directory):
    """Lee únicamente los archivos locales validados; no necesita Internet."""
    paths = {name: Path(directory) / name for name in FILES}
    for name, expected in FILES.items():
        path = paths[name]
        if not path.is_file():
            raise FileNotFoundError(f"Falta {path}; ejecuta python3 -m caso download")
        if md5_file(path) != expected:
            raise ValueError(f"Checksum incorrecto: {path}; repite python3 -m caso download")
    train_images = read_images(paths["train-images-idx3-ubyte.gz"])
    train_labels = read_labels(paths["train-labels-idx1-ubyte.gz"])
    test_images = read_images(paths["t10k-images-idx3-ubyte.gz"])
    test_labels = read_labels(paths["t10k-labels-idx1-ubyte.gz"])
    if (train_images.shape != (60000, 28, 28) or train_labels.shape != (60000,)
            or test_images.shape != (10000, 28, 28) or test_labels.shape != (10000,)):
        raise ValueError("Fashion-MNIST local no tiene las particiones oficiales")
    return (train_images, train_labels), (test_images, test_labels)
