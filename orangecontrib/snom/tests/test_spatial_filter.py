import unittest
import numpy as np
from AnyQt.QtWidgets import QApplication

fromorangecontrib.snom.widgets.preprocessors.spatial_filter import SpatialFilterProcessor, SpatialFilterEditor

app = QApplication.instance() or QApplication([])

class TestSpatialFilterProcessor(unittest.TestCase):
    def setUp(self):
        # 5x5 image with outlier pixel at centre-up position 
        self.test_image = np.ones((5, 5))
        self.test_image[2, 2] = 100.0 
        self.mock_data = None

    def test_median_filter_removes_outlier(self):
        # Meidan filter should remove outliers
        processor = SpatialFilterProcessor(method="median", size=3)
        result = processor.transform_image(self.test_image, self.mock_data)
        
        # The value 100.0 should have been substituted by 1.0
        self.assertEqual(result[2, 2], 1.0)
        self.assertTrue(np.all(result == 1.0))

    def test_gaussian_filter(self):
        processor = SpatialFilterProcessor(method="gaussian", sigma=1.0)
        result = processor.transform_image(self.test_image, self.mock_data)
        
        # Gaussian blurs the values, so the centre should be lower than 100 but higher than 1
        self.assertLess(result[2, 2], 100.0)
        self.assertGreater(result[2, 2], 1.0)

class TestSpatialFilterEditor(unittest.TestCase):
    def setUp(self):
        self.editor = SpatialFilterEditor(None)

    def test_ui_updates_on_method_change(self):
        # Test for UI logical interface
        self.editor.setParameters({"method": "gaussian", "sigma": 2.0})
        self.assertTrue(self.editor.sigma_edit.isEnabled())
        self.assertFalse(self.editor.size_edit.isEnabled())
        
        self.editor.setParameters({"method": "median", "size": 3})
        self.assertFalse(self.editor.sigma_edit.isEnabled())
        self.assertTrue(self.editor.size_edit.isEnabled())

if __name__ == "__main__":
    unittest.main()