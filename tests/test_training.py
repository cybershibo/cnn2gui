import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.datasets import load_iris

from proyecto.training import load_run, predict, prepare, train


class TrainingTests(unittest.TestCase):
    def setUp(self):
        iris = load_iris(as_frame=True)
        self.frame = iris.data.copy()
        self.frame['class'] = iris.target

    def test_training_persistence_and_inference(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, NN2GUI_DATA=directory):
            run = train(self.frame, 'class', [16], 'ReLU', 20, 0.01, 16, 42)
            config, metrics, history = load_run(run)
            splits = [set(config['split'][key]) for key in ('train', 'validation', 'test')]
            self.assertFalse(splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2])
            self.assertEqual(len(set.union(*splits)), len(self.frame))
            np.testing.assert_allclose(config['mean'], self.frame.iloc[config['split']['train'], :-1].mean())
            probabilities = predict(run, self.frame.iloc[0, :-1].tolist())
            self.assertAlmostEqual(float(probabilities.sum()), 1, places=5)
            np.testing.assert_array_equal(probabilities, predict(run, self.frame.iloc[0, :-1].tolist()))
            self.assertGreater(metrics['accuracy'], 0.5)
            self.assertEqual(len(history), 20)

    def test_reject_incompatible_data(self):
        for replacement in [float('nan'), float('inf'), 'texto']:
            frame = self.frame.copy().astype(object)
            frame.iloc[0, 0] = replacement
            with self.assertRaises(ValueError):
                prepare(frame, 'class')

    def test_reject_path_as_run_id(self):
        with self.assertRaises(ValueError):
            load_run('../weights.pt')


if __name__ == '__main__':
    unittest.main()
