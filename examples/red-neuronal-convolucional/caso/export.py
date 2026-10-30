"""Exporta ejemplos variados del test oficial sin alterar sus píxeles."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .dataset import load_dataset
from .pipeline import CLASSES


SLUGS = ("camiseta", "pantalon", "sueter", "vestido", "abrigo",
         "sandalia", "camisa", "tenis", "bolsa", "botin")


def export_examples(data, out, per_class=5, seed=42, skip_index=0):
    """Genera PNG originales del test, CSV de etiquetas y catálogo ampliado."""
    if not 1 <= per_class <= 100:
        raise ValueError("--per-class debe estar entre 1 y 100.")
    dest = Path(out)
    if dest.exists():
        raise FileExistsError(f"La carpeta {dest} ya existe; elige otra ruta con --out.")

    (_, _), (images, labels) = load_dataset(data)
    rng = np.random.default_rng(seed)
    selections = []
    for class_id in range(len(CLASSES)):
        available = np.flatnonzero((labels == class_id) & (np.arange(len(labels)) != skip_index))
        if len(available) < per_class:
            raise ValueError(f"La clase {CLASSES[class_id]} no tiene {per_class} ejemplos disponibles.")
        selections.append(rng.choice(available, size=per_class, replace=False))

    dest.mkdir(parents=True)
    tile_width, tile_height = 120, 144
    catalog = Image.new("RGB", (tile_width * len(CLASSES), tile_height * per_class), "white")
    painter = ImageDraw.Draw(catalog)
    with (dest / "etiquetas.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("archivo", "indice_test", "etiqueta_id", "etiqueta"))
        for class_id, indices in enumerate(selections):
            for row, index in enumerate(indices):
                index = int(index)
                filename = f"test_{index:05d}_{SLUGS[class_id]}.png"
                original = Image.fromarray(images[index])
                original.save(dest / filename)
                writer.writerow((filename, index, class_id, CLASSES[class_id]))

                x, y = class_id * tile_width, row * tile_height
                catalog.paste(original.resize((112, 112), Image.Resampling.NEAREST), (x + 4, y + 24))
                painter.text((x + 4, y + 4), f"{index}: {CLASSES[class_id]}", fill="black")
    catalog.save(dest / "catalogo.png")
    print(f"Exportadas {len(CLASSES) * per_class} imágenes del test en {dest}")
    print(f"Etiquetas: {dest / 'etiquetas.csv'}; vista previa: {dest / 'catalogo.png'}")
