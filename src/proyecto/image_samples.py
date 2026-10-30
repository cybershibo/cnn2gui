"""Selecciones reproducibles del test oficial, sin duplicar los archivos del dataset."""
import json
from pathlib import Path
import secrets
from uuid import uuid4

import numpy as np

from proyecto.cnn import CATALOG, data_root

COUNT = 5000


def sample_indices(total, seed):
    if total < COUNT:
        raise ValueError('El conjunto de prueba debe tener al menos 5 000 imágenes.')
    return np.random.default_rng(seed).choice(total, size=COUNT, replace=False).tolist()


def save_selection(dataset, total):
    if dataset not in CATALOG:
        raise ValueError('Dataset no compatible.')
    seed = secrets.randbits(32)
    selection = dict(dataset=dataset, partition='official_test', seed=seed,
                     total=total, indices=sample_indices(total, seed))
    directory = data_root()/'image-selections'/dataset
    directory.mkdir(parents=True, exist_ok=True)
    identifier = uuid4().hex
    (directory/f'{identifier}.json').write_text(json.dumps(selection))
    return identifier


def load_selection(dataset, identifier):
    if dataset not in CATALOG or len(identifier) != 32 or any(c not in '0123456789abcdef' for c in identifier):
        raise ValueError('Selección inválida.')
    selection = json.loads((data_root()/'image-selections'/dataset/f'{identifier}.json').read_text())
    indices = selection['indices']
    if (selection['dataset'] != dataset or selection['partition'] != 'official_test'
            or len(indices) != COUNT or len(set(indices)) != COUNT
            or any(type(i) is not int or not 0 <= i < selection['total'] for i in indices)):
        raise ValueError('La selección no contiene 5 000 imágenes únicas válidas.')
    return selection


def saved_selections(dataset):
    directory = data_root()/'image-selections'/dataset
    return [p.stem for p in sorted(directory.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)]
