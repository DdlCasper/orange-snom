import unittest
import numpy as np
from AnyQt.QtWidgets import QApplication

from  orangecontrib.snom.widgets.preprocessors.gradient_watershed import SpectralWatershedProcessor, SpectralWatershedEditor

app = QApplication.instance() or QApplication([])

class TestSpectralWatershedProcessor(unittest.TestCase):
    def setUp(self):
        # Create a blob as 2 locked circles to simulate adjacent cells
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0]
        ], dtype=float)
        
        # We create a fake gradient to force the cute in the centre of the blob
        self.mock_gradient = np.zeros_like(self.test_image)
        self.mock_gradient[:, 2:4] = 1.0 # Bordo forte in mezzo

    def test_apply_spectral_watershed(self):
        processor = SpectralWatershedProcessor(min_distance=1)
        # We give the simulated gradient
        result = processor.apply_spectral_watershed(self.test_image, self.mock_gradient)
        
        # Control if pixels (1,1) and (1,4) belong to different labels
        label_left = result[1, 1]
        label_right = result[1, 4]
        
        self.assertNotEqual(label_left, label_right)
        self.assertGreater(label_left, 0)
        self.assertGreater(label_right, 0)

class TestSpectralWatershedEditor(unittest.TestCase):
    def test_set_parameters(self):
        editor = SpectralWatershedEditor(None)
        params = {"min_distance": 15, "grad_sigma": 2.5}
        editor.setParameters(params)
        
        self.assertEqual(editor.min_distance, 15)[cite: 2]
        self.assertEqual(editor.grad_sigma, 2.5)[cite: 2]

if __name__ == "__main__":
    unittest.main()