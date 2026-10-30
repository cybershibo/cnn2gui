import io
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

from proyecto.cnn import build_model, load_dataset, predict_cnn, prepare_image, prepare_pixels, train_cnn
from proyecto.training import load_run


def fixture():
    rng = np.random.default_rng(42)
    return ((rng.integers(0, 256, (240, 28, 28), dtype=np.uint8), np.tile(np.arange(10), 24)),
            (rng.integers(0, 256, (20, 28, 28), dtype=np.uint8), np.tile(np.arange(10), 2)))


class CNNTests(unittest.TestCase):
    def test_preprocessing_matches_example(self):
        image = fixture()[1][0][0]
        stream = io.BytesIO()
        Image.fromarray(image).save(stream, format='PNG')
        np.testing.assert_array_equal(prepare_pixels(image), prepare_image(stream.getvalue()))
        self.assertEqual(prepare_pixels(image).shape, (1, 28, 28, 1))

    def test_model_shapes_and_validation(self):
        for filters in ([8], [8, 16], [8, 16, 32]):
            model = build_model(filters, 16, 0.2, 0.001)
            self.assertEqual(model.output_shape, (None, 10))
        with self.assertRaises(ValueError):
            load_dataset('iris')
        with self.assertRaises(ValueError):
            build_model([7], 16, 0.2, 0.001)

    def test_training_and_saved_model(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, NN2GUI_DATA=directory), patch('proyecto.cnn.load_dataset', return_value=fixture()):
            run = train_cnn('fashion_mnist', filters=[8], dense=8, epochs=1, sample_count=200)
            config, metrics, history = load_run(run)
            self.assertFalse(set(config['split']['train']) & set(config['split']['validation']))
            self.assertEqual(config['sample_count'], 200)
            self.assertEqual(config['test_partition'], 'official_test')
            self.assertEqual(sum(map(sum, metrics['confusion'])), 20)
            self.assertEqual(len(history), 1)
            from pathlib import Path
            content = (Path(directory) / run / 'sample.png').read_bytes()
            first = predict_cnn(run, content)
            np.testing.assert_allclose(first, predict_cnn(run, content))
            self.assertAlmostEqual(float(first.sum()), 1, places=5)
            app = AppTest.from_file('src/proyecto/app.py', default_timeout=60).run()
            self.assertFalse(app.exception)
            next(item for item in app.button if item.label == 'Clasificar imagen').click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any('Clase predicha:' in item.value for item in app.success))

    def test_catalog_selection_does_not_download_automatically(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, NN2GUI_DATA=directory), patch('proyecto.cnn_ui.load_dataset', return_value=fixture()) as loader:
            app = AppTest.from_file('src/proyecto/app.py', default_timeout=60).run()
            self.assertFalse(app.exception)
            select = next(item for item in app.selectbox if item.label == 'Dataset público y gratuito')
            select.select('mnist').run()
            self.assertFalse(app.exception)
            loader.assert_not_called()
            next(item for item in app.button if item.label == 'Descargar y explorar dataset').click().run()
            loader.assert_called_once_with('mnist')
            self.assertFalse(app.exception)


if __name__ == '__main__':
    unittest.main()
