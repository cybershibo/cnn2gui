"""Verifica que los PNG exportados sean muestras etiquetadas del test."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from caso.export import export_examples


class ExportTest(unittest.TestCase):
    def test_balanced_examples_preserve_pixels_and_do_not_overwrite(self):
        labels = np.repeat(np.arange(10, dtype=np.uint8), 4)
        images = np.stack([np.full((28, 28), index, dtype=np.uint8)
                           for index in range(len(labels))])
        with TemporaryDirectory() as folder:
            dest = Path(folder) / "ejemplos"
            with patch("caso.export.load_dataset", return_value=((None, None), (images, labels))):
                export_examples("fake", dest, per_class=2, seed=42, skip_index=0)
                with self.assertRaises(FileExistsError):
                    export_examples("fake", dest, per_class=2, seed=42, skip_index=0)
            with (dest / "etiquetas.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 20)
            self.assertEqual(len({row["indice_test"] for row in rows}), 20)
            self.assertNotIn("0", [row["indice_test"] for row in rows])
            for class_id in range(10):
                self.assertEqual(sum(int(row["etiqueta_id"]) == class_id for row in rows), 2)
            for row in rows:
                with Image.open(dest / row["archivo"]) as image:
                    self.assertEqual(image.size, (28, 28))
                    np.testing.assert_array_equal(np.asarray(image), images[int(row["indice_test"])])
            with Image.open(dest / "catalogo.png") as catalog:
                self.assertEqual(catalog.size, (1200, 288))


if __name__ == "__main__":
    unittest.main()
