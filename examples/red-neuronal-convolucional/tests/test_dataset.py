"""Descarga recuperable y lectura de los archivos oficiales IDX."""

import gzip
import hashlib
import io
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from caso import dataset


class DatasetTest(unittest.TestCase):
    def test_download_uses_second_source_then_verified_cache(self):
        data = b"dataset-de-ensayo"
        checksum = hashlib.md5(data, usedforsecurity=False).hexdigest()
        with tempfile.TemporaryDirectory() as root:
            with patch.object(dataset, "urlopen", side_effect=[OSError("caído"), io.BytesIO(data)]) as get:
                path = dataset.download_one(root, "train.gz", checksum,
                                            bases=("https://primary/", "https://backup/"))
                self.assertEqual(get.call_count, 2)
            self.assertEqual(path.read_bytes(), data)
            with patch.object(dataset, "urlopen") as get:
                self.assertEqual(dataset.download_one(root, "train.gz", checksum), path)
                get.assert_not_called()

    def test_idx_readers_validate_images_labels_and_corruption(self):
        with tempfile.TemporaryDirectory() as root:
            images = Path(root) / "images.gz"
            labels = Path(root) / "labels.gz"
            pixels = np.arange(28 * 28, dtype=np.uint8).tobytes()
            with gzip.open(images, "wb") as stream:
                stream.write(struct.pack(">IIII", 2051, 1, 28, 28) + pixels)
            with gzip.open(labels, "wb") as stream:
                stream.write(struct.pack(">II", 2049, 1) + bytes([9]))
            self.assertEqual(dataset.read_images(images).shape, (1, 28, 28))
            self.assertEqual(dataset.read_labels(labels).tolist(), [9])
            with gzip.open(labels, "wb") as stream:
                stream.write(struct.pack(">II", 2049, 1) + bytes([10]))
            with self.assertRaisesRegex(ValueError, "fuera"):
                dataset.read_labels(labels)


if __name__ == "__main__":
    unittest.main()
