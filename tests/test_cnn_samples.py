import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest
from proyecto.image_samples import sample_indices, save_selection, load_selection
from proyecto.image_samples_ui import test_images


class ImageSelectionTests(unittest.TestCase):
    def test_exact_unique_reproducible_sample(self):
        first=sample_indices(10000,42)
        self.assertEqual(len(first),5000)
        self.assertEqual(len(set(first)),5000)
        self.assertEqual(first,sample_indices(10000,42))
        self.assertNotEqual(first,sample_indices(10000,43))
        self.assertTrue(all(0 <= i < 10000 for i in first))
        with self.assertRaises(ValueError):
            sample_indices(4999,42)

    def test_selection_persists_and_rejects_invalid_ids(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, NN2GUI_DATA=directory):
            identifier=save_selection('mnist',10000)
            selection=load_selection('mnist',identifier)
            self.assertEqual(selection['partition'],'official_test')
            self.assertEqual(selection['indices'],sample_indices(10000,selection['seed']))
            self.assertEqual(selection,load_selection('mnist',identifier))
            with self.assertRaises((ValueError,FileNotFoundError)):
                load_selection('fashion_mnist',identifier)
            with self.assertRaises(ValueError):
                load_selection('mnist','../../file')

    def test_gallery_chooses_original_test_pixels(self):
        images=np.zeros((10000,28,28),dtype=np.uint8)
        images[:,0,0]=np.arange(10000)%256
        labels=np.arange(10000)%10
        test_images.clear()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, NN2GUI_DATA=directory), patch('proyecto.image_samples_ui.load_dataset',return_value=((np.zeros((1,28,28),dtype=np.uint8),np.array([0])),(images,labels))) as loader:
            script=Path(directory)/'gallery.py'
            script.write_text("import streamlit as st\nfrom proyecto.image_samples_ui import select_dataset_image\ncontent,label=select_dataset_image('fashion_mnist')\nst.session_state['chosen_content']=content\nst.session_state['chosen_label']=label\n")
            app=AppTest.from_file(str(script),default_timeout=60).run()
            loader.assert_not_called()
            next(x for x in app.button if x.label=='Extraer 5 000 imágenes al azar').click().run()
            self.assertFalse(app.exception)
            identifier=app.session_state['image-selection-fashion_mnist']
            indices=load_selection('fashion_mnist',identifier)['indices']
            next(x for x in app.number_input if x.label=='Página de la galería').set_value(2).run()
            next(x for x in app.button if x.label=='Elegir #21').click().run()
            self.assertFalse(app.exception)
            actual=np.asarray(Image.open(io.BytesIO(app.session_state['chosen_content'])))
            np.testing.assert_array_equal(actual,images[indices[20]])
            self.assertEqual(app.session_state['chosen_label'],int(labels[indices[20]]))
            app.run()
            self.assertEqual(app.session_state['image-selection-fashion_mnist'],identifier)
            self.assertEqual(len(list((Path(directory)/'image-selections'/'fashion_mnist').glob('*.json'))),1)
        test_images.clear()

if __name__ == '__main__':
    unittest.main()
