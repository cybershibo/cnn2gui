import unittest
from unittest.mock import patch
from proyecto.compute import validate_device, probe

class ComputeTests(unittest.TestCase):
    def test_device_validation(self):
        for device in ['cpu', 'cuda:0', 'cuda:2']:
            validate_device(device)
        for device in ['cuda', 'gpu', 'cuda:-1', '/GPU:0']:
            with self.assertRaises(ValueError):
                validate_device(device)

    def test_unavailable_gpu_is_reported(self):
        with patch('proyecto.compute.worker', side_effect=RuntimeError('driver no disponible')):
            devices, reason = probe('torch')
        self.assertEqual(devices, [])
        self.assertIn('driver no disponible', reason)
