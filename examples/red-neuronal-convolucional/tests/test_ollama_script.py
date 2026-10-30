"""Comprueba la conexión entre Docker, la predicción y la API de Ollama."""

import io
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import explicar_imagen


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class OllamaScriptTest(unittest.TestCase):
    def test_full_flow_sends_prediction_without_image_pixels(self):
        prediction = {"imagen": "/imagen/foto.png", "prediccion": "bolsa",
                      "probabilidades": {"bolsa": 0.9, "camisa": 0.08, "abrigo": 0.02}}
        http_requests = []
        commands = []

        def fake_docker(command, **kwargs):
            commands.append(command)
            self.assertTrue(kwargs["check"])
            return subprocess.CompletedProcess(command, 0, json.dumps(prediction), "")

        def fake_urlopen(req, timeout):
            http_requests.append((req, timeout))
            data = ({"models": [{"name": "modelo-local:latest"}]}
                    if req.full_url.endswith("/api/tags") else {"response": "Parece una bolsa."})
            return Response(json.dumps(data).encode("utf-8"))

        with TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "outputs" / "moda-01"
            run.mkdir(parents=True)
            (run / "modelo.keras").touch()
            (run / "metadatos.json").write_text("{}", encoding="utf-8")
            photo = root / "foto.png"
            photo.write_bytes(b"contenido-privado-de-imagen")
            with patch.object(explicar_imagen, "ROOT", root), \
                    patch.object(explicar_imagen.subprocess, "run", side_effect=fake_docker), \
                    patch.object(explicar_imagen.request, "urlopen", side_effect=fake_urlopen), \
                    patch("sys.stdout", new_callable=io.StringIO) as stdout:
                explicar_imagen.main([str(photo), "--question", "¿Qué es?"])

        self.assertNotIn("--gpus", commands[0])
        self.assertIn(f"{root}:/app:ro", commands[0])
        self.assertIn("/imagen/foto.png", commands[0])
        self.assertEqual(http_requests[0][0].full_url, "http://127.0.0.1:11434/api/tags")
        sent = json.loads(http_requests[1][0].data)
        self.assertEqual(sent["model"], "modelo-local:latest")
        self.assertFalse(sent["stream"])
        self.assertIn("bolsa", sent["prompt"])
        self.assertIn("¿Qué es?", sent["prompt"])
        self.assertNotIn("contenido-privado-de-imagen", sent["prompt"])
        self.assertIn("Parece una bolsa.", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
