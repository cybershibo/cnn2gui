import os
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from proyecto.training import load_run


class AppTests(unittest.TestCase):
    def test_train_and_predict(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, NN2GUI_DATA=directory):
            app = AppTest.from_file('src/proyecto/app.py', default_timeout=60).run()
            self.assertFalse(app.exception)
            self.assertEqual([tab.label for tab in app.tabs], ['1 · Arquitectura', '2 · Entrenamiento', '3 · Ejecución'])
            self.assertFalse(app.tabs[0].button)
            self.assertFalse(app.tabs[2].button)
            self.assertTrue(any(item.label == 'Capas ocultas' for item in app.tabs[0].slider))
            self.assertTrue(any(item.label.startswith('Épocas:') for item in app.tabs[1].slider))
            next(item for item in app.slider if item.label == 'Neuronas en capa 1').set_value(64).run()
            next(item for item in app.tabs[1].button if item.label == 'Entrenar y guardar').click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.metric)
            run = app.session_state['last_run']
            self.assertEqual(load_run(run)[0]['hidden'], [64, 32])
            self.assertTrue(any(item.label == 'Predecir' for item in app.tabs[2].button))
            self.assertFalse(any(item.label == 'Predecir' for item in app.tabs[1].button))
            next(item for item in app.slider if item.label == 'Neuronas en capa 1').set_value(16).run()
            next(item for item in app.checkbox if item.label == 'Visualizar activaciones en tiempo real').check().run()
            next(item for item in app.slider if item.label == 'Pausa didáctica entre capas (segundos)').set_value(0.0).run()
            next(item for item in app.tabs[2].button if item.label == 'Predecir').click().run()
            self.assertEqual(load_run(run)[0]['hidden'], [64, 32])
            self.assertFalse(app.exception)
            self.assertTrue(any('Clase predicha:' in item.value for item in app.success))
            self.assertTrue(any(item.label == 'Inspeccionar capa calculada' for item in app.select_slider))
            app.select_slider[0].set_value(0).run()
            self.assertFalse(app.exception)
            self.assertTrue(any('Entrada normalizada' in item.value for item in app.markdown))


if __name__ == '__main__':
    unittest.main()
