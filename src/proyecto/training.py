"""Motor de clasificación tabular; independiente de Streamlit."""
from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn


def prepare(frame: pd.DataFrame, target: str):
    if not frame.columns.is_unique or target not in frame:
        raise ValueError("Selecciona una columna objetivo y nombres de columna únicos.")
    x = frame.drop(columns=[target])
    if x.shape[1] == 0 or len(frame) < 30:
        raise ValueError("Se necesitan al menos 30 filas y una columna de entrada.")
    if x.shape[1] > 100 or len(frame) > 100000:
        raise ValueError("Esta versión admite hasta 100 entradas y 100 000 filas.")
    if frame.isna().any().any():
        raise ValueError("Hay valores vacíos: completa o elimina esas filas antes de importar.")
    try:
        values = x.to_numpy(dtype=np.float32)
    except (ValueError, TypeError) as exc:
        raise ValueError("Todas las entradas deben ser numéricas; el objetivo puede ser texto.") from exc
    if not np.isfinite(values).all():
        raise ValueError("Las entradas deben ser números finitos.")
    labels = LabelEncoder().fit(frame[target].astype(str))
    y = labels.transform(frame[target].astype(str))
    if not 2 <= len(labels.classes_) <= 20 or np.bincount(y).min() < 10:
        raise ValueError("Usa de 2 a 20 clases, con al menos 10 filas por clase.")
    return values, y, list(x.columns), labels.classes_.tolist()


def build_model(inputs, hidden, outputs, activation):
    layers = []
    for width in hidden:
        layers.extend([nn.Linear(inputs, width), nn.ReLU() if activation == "ReLU" else nn.Tanh()])
        inputs = width
    return nn.Sequential(*layers, nn.Linear(inputs, outputs))


def train(frame, target, hidden, activation, epochs, learning_rate, batch_size, seed, progress=None):
    if not hidden or len(hidden) > 4 or any(not 1 <= n <= 256 for n in hidden):
        raise ValueError("Usa de 1 a 4 capas ocultas con 1 a 256 neuronas.")
    if activation not in ("ReLU", "Tanh") or not 1 <= epochs <= 300:
        raise ValueError("Activación o número de épocas no admitido.")
    if not 0 < learning_rate <= 0.1 or not 1 <= batch_size <= 512:
        raise ValueError("Tasa de aprendizaje o lote no admitido.")
    x, y, columns, classes = prepare(frame, target)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    indices = np.arange(len(y))
    fit, test = train_test_split(indices, test_size=0.2, stratify=y, random_state=seed)
    fit, val = train_test_split(fit, test_size=0.25, stratify=y[fit], random_state=seed)
    scaler = StandardScaler().fit(x[fit])
    x = torch.tensor(scaler.transform(x), dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    model = build_model(len(columns), hidden, len(classes), activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        model.train()
        order = torch.tensor(fit)[torch.randperm(len(fit))]
        for batch in order.split(batch_size):
            optimizer.zero_grad()
            loss = loss_fn(model(x[batch]), y[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = loss_fn(model(x[fit]), y[fit]).item()
            val_loss = loss_fn(model(x[val]), y[val]).item()
        if not np.isfinite([train_loss, val_loss]).all():
            raise ValueError("El entrenamiento divergió. Prueba una tasa de aprendizaje menor.")
        history.append({"época": epoch + 1, "entrenamiento": train_loss, "validación": val_loss})
        if progress:
            progress(history)
    with torch.no_grad():
        predicted = model(x[test]).argmax(1).numpy()
    config = dict(columns=columns, classes=classes, hidden=hidden, activation=activation,
                  mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), target=target,
                  epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, seed=seed,
                  split={"train": fit.tolist(), "validation": val.tolist(), "test": test.tolist()},
                  versions={"torch": str(torch.__version__), "numpy": np.__version__})
    metrics = dict(accuracy=accuracy_score(y[test].numpy(), predicted),
                   confusion=confusion_matrix(y[test].numpy(), predicted,
                                             labels=list(range(len(classes)))).tolist())
    directory = Path(os.environ.get("NN2GUI_DATA", "data")) / uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), directory / "weights.pt")
    frame.to_csv(directory / "dataset.csv", index=False)
    for name, value in [("config", config), ("metrics", metrics), ("history", history)]:
        (directory / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2))
    return directory.name


def load_run(run_id):
    if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
        raise ValueError("Identificador de ejecución inválido.")
    directory = Path(os.environ.get("NN2GUI_DATA", "data")) / run_id
    return [json.loads((directory / f"{name}.json").read_text())
            for name in ("config", "metrics", "history")]


def predict(run_id, values):
    config, _, _ = load_run(run_id)
    x = np.asarray(values, dtype=np.float32)
    if x.shape != (len(config["columns"]),) or not np.isfinite(x).all():
        raise ValueError("Las entradas no coinciden con el modelo o no son finitas.")
    x = (x - np.asarray(config["mean"])) / np.asarray(config["scale"])
    model = build_model(len(x), config["hidden"], len(config["classes"]), config["activation"])
    directory = Path(os.environ.get("NN2GUI_DATA", "data")) / run_id
    model.load_state_dict(torch.load(directory / "weights.pt", map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).softmax(0).numpy()
