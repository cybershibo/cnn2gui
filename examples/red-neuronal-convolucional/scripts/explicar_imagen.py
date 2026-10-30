#!/usr/bin/env python3
"""Clasifica una imagen con la CNN guardada y explica el resultado con Ollama.

Solo requiere Python estándar en Ubuntu, Docker y una instalación local de Ollama.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib import error, request


ROOT = Path(__file__).resolve().parents[1]
DOCKER_IMAGE = "armando-cnn:tf215"


def docker_prediction(image, run, docker_image=DOCKER_IMAGE):
    """Ejecuta la CNN en CPU y devuelve la salida JSON del comando predict."""
    image = Path(image).expanduser().resolve()
    if not image.is_file():
        raise ValueError(f"No existe el archivo de imagen: {image}")
    run = Path(run).expanduser()
    run = (ROOT / run).resolve() if not run.is_absolute() else run.resolve()
    try:
        run_in_container = run.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("La carpeta --run debe estar dentro del repositorio.") from exc
    for name in ("modelo.keras", "metadatos.json"):
        if not (run / name).is_file():
            raise ValueError(f"Falta {run / name}; ejecuta primero el entrenamiento.")

    command = [
        "docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp", "-v", f"{ROOT}:/app:ro",
        "-v", f"{image.parent}:/imagen:ro", "-w", "/app", docker_image,
        "python", "-m", "caso", "predict", "--run",
        f"/app/{run_in_container}", "--image", f"/imagen/{image.name}",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("No se encontró Docker. Instálalo y comprueba 'docker --version'.") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"Falló la predicción en Docker (código {exc.returncode}):\n{detail}") from exc
    try:
        prediction = json.loads(result.stdout)
        if not isinstance(prediction["prediccion"], str) or not isinstance(prediction["probabilidades"], dict):
            raise ValueError("Campos de predicción inválidos")
        return prediction
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"La CNN no devolvió una predicción JSON válida:\n{result.stdout}") from exc


def ollama_json(url, payload=None, timeout=300):
    """Llama a la API HTTP local de Ollama con una respuesta JSON no fragmentada."""
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama devolvió HTTP {exc.code}: {detail}") from exc
    except (error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"No se pudo conectar con Ollama en {url}: {exc}") from exc


def select_model(base_url, specified=None):
    if specified:
        return specified
    response = ollama_json(f"{base_url}/api/tags", timeout=10)
    names = [item.get("name") for item in response.get("models", []) if item.get("name")]
    if not names:
        raise RuntimeError("Ollama no tiene modelos descargados. Ejecuta 'ollama list' y luego 'ollama pull NOMBRE'.")
    return names[0]


def build_prompt(prediction, question):
    ranked = sorted(prediction["probabilidades"].items(), key=lambda item: -item[1])
    probabilities = ", ".join(f"{name}: {score:.1%}" for name, score in ranked[:3])
    return (
        "Responde en español, de manera breve y clara. Actúas como intérprete de un "
        "clasificador CNN entrenado únicamente con Fashion-MNIST (prendas 28×28 en gris). "
        "No has visto la imagen: solo recibes el resultado numérico de la CNN. "
        "No inventes detalles visuales ni afirmes que se trata con certeza de una prenda. "
        "Si la entrada es una foto real, explica que puede estar fuera del dominio de entrenamiento. "
        "Las salidas softmax no son garantías de exactitud ni probabilidades calibradas.\n\n"
        f"Categoría predicha: {prediction['prediccion']}.\n"
        f"Tres salidas softmax más altas: {probabilities}.\n"
        f"Petición del usuario: {question}\n"
        "Incluye la categoría predicha y una nota breve sobre la limitación del modelo."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="CNN Fashion-MNIST + respuesta en español con Ollama")
    parser.add_argument("imagen", help="Ruta de un PNG, JPG u otra imagen compatible con Pillow")
    parser.add_argument("--run", default="outputs/moda-01", help="Carpeta del modelo entrenado dentro del repositorio")
    parser.add_argument("--model", help="Nombre exacto de un modelo local de 'ollama list'; por defecto el primero")
    parser.add_argument("--question", default="¿Qué resultado obtuvo el clasificador?",
                        help="Pregunta que se incluye en el prompt de Ollama")
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
                        help="URL base de Ollama en Ubuntu")
    parser.add_argument("--docker-image", default=DOCKER_IMAGE, help="Imagen Docker construida en el README")
    args = parser.parse_args(argv)
    base_url = args.ollama_url.rstrip("/")
    if "://" not in base_url:
        base_url = "http://" + base_url
    if not base_url.startswith(("http://", "https://")):
        parser.error("--ollama-url debe ser una URL HTTP o HTTPS")

    try:
        prediction = docker_prediction(args.imagen, args.run, args.docker_image)
        model = select_model(base_url, args.model)
        answer = ollama_json(f"{base_url}/api/generate", {
            "model": model, "prompt": build_prompt(prediction, args.question), "stream": False,
        })["response"].strip()
        if not answer:
            raise RuntimeError("Ollama respondió sin texto.")
    except (RuntimeError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    top = sorted(prediction["probabilidades"].items(), key=lambda item: -item[1])[:3]
    print(f"Imagen: {Path(args.imagen).expanduser().resolve()}")
    print(f"Predicción CNN: {prediction['prediccion']}")
    print("Salidas principales: " + ", ".join(f"{name} {score:.1%}" for name, score in top))
    print(f"\nRespuesta de Ollama ({model}):\n{answer}")


if __name__ == "__main__":
    main()
