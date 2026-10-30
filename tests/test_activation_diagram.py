import base64
import io
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from proyecto.network_diagram import bitmap, network_svg


class DiagramTests(unittest.TestCase):
    def trace(self):
        return [dict(name=name, values=np.full(shape, 0.25, dtype=np.float32))
                for name, shape in [('Entrada normalizada', (28, 28, 1)),
                                    ('1 · Conv2D', (26, 26, 8)), ('2 · MaxPooling2D', (13, 13, 8)),
                                    ('3 · Flatten', (1352,)), ('4 · Dense', (16,)),
                                    ('5 · Dropout', (16,)), ('6 · Dense', (10,))]]

    def test_cnn_structure_and_real_probabilities(self):
        trace = self.trace()
        trace[-1]['values'] = np.array([0.55]+[0.05]*9)
        diagram = network_svg(trace, 1, dict(filters=[8], classes=['clase <uno>']+[str(i) for i in range(9)]))
        svg = ET.fromstring(diagram[diagram.index('<svg'):diagram.index('</svg>')+6])
        content = ''.join(svg.itertext())
        for label in ['ENTRADA', 'SALIDA', 'Convolución', 'Agrupación', 'Aplanamiento',
                      'Extracción de características', 'Clasificación', '55.0%', 'clase <uno>']:
            self.assertIn(label, content)
        self.assertIn('#d99b13', diagram)
        self.assertIn('data:image/png;base64,', diagram)
        self.assertGreater(len(svg.findall('.//{http://www.w3.org/2000/svg}circle')), 10)

    def test_partial_trace_has_no_invented_output(self):
        diagram = network_svg(self.trace()[:3], 2, dict(filters=[8]))
        self.assertIn('Esperando la capa de salida', diagram)
        self.assertNotIn('Predicción:', diagram)

    def test_input_pixels_keep_absolute_intensity(self):
        uri = bitmap(np.full((2, 2), 0.25), scale=1.0)
        pixels = np.asarray(Image.open(io.BytesIO(base64.b64decode(uri.split(',')[1]))))
        np.testing.assert_array_equal(pixels, np.full((2, 2), 63))

    def test_dense_diagram(self):
        trace = [dict(name='Entrada normalizada', values=np.array([-1., 1.])),
                 dict(name='1 · Linear', values=np.array([-0.5, 0.5])),
                 dict(name='2 · ReLU', values=np.array([0., 0.5])),
                 dict(name='Salida · probabilidades', values=np.array([0.2, 0.8]))]
        diagram = network_svg(trace, 1, dict(classes=['a', 'b']))
        self.assertIn('RED NEURONAL DENSA', diagram)
        self.assertIn('80.0%', diagram)
        self.assertIn('1 · Linear', diagram)


if __name__ == '__main__':
    unittest.main()
