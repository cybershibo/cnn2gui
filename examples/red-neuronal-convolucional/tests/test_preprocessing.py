"""Comprueba que el PNG de demostración recorre el mismo escalado que los datos."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from caso.pipeline import prepare_file, prepare_pixels


class InputParityTest(unittest.TestCase):
    def test_exported_png_and_original_array_produce_identical_model_input(self):
        original = np.arange(28 * 28, dtype=np.uint8).reshape(28, 28)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "entrada.png"
            Image.fromarray(original).save(path)
            np.testing.assert_array_equal(prepare_pixels(original), prepare_file(path))
            self.assertEqual(prepare_file(path).shape, (1, 28, 28, 1))
            self.assertEqual(float(prepare_file(path).max()), float(original.max() / 255.0))


if __name__ == "__main__":
    unittest.main()
