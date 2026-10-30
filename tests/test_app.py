import os
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    def test_train_and_predict(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, NN2GUI_DATA=directory):
            app = AppTest.from_file('src/proyecto/app.py', default_timeout=60).run()
            self.assertFalse(app.exception)
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.metric)
            app.button[1].click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any('Clase predicha:' in item.value for item in app.success))


if __name__ == '__main__':
    unittest.main()
