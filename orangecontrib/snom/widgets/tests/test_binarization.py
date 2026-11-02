import unittest
import numpy as np
from AnyQt.QtWidgets import QApplication

from orangecontrib.snom.widgets.preprocessors.binarization import BinarizationProcessor, BinarizationEditor

app = QApplication.instance()
if app is None:
    app = QApplication([])

class TestBinarization(unittest.TestCase):

    def setUp(self):
        # We setup the data and see if NaN are correctly handled 
        self.mock_image = np.array([
            [0.1, 0.9, 0.4],
            [np.nan, 0.6, 0.2]
        ])
        
        self.mock_data = None 

    # Test main logic

    def test_processor_manual_threshold(self):
        """Tests binarization with the method   'manual'."""
        processor = BinarizationProcessor(method="manual", manual_threshold=0.5) 
        
        result = processor.transform_image(self.mock_image, self.mock_data)
        
        # Verify if values above 0.5 are 1 and below 0.5 are 0.0
        self.assertEqual(result[0, 0], 0.0) # 0.1 < 0.5
        self.assertEqual(result[0, 1], 1.0) # 0.9 > 0.5
        
        # Verify if it's Nan
        self.assertTrue(np.isnan(result[1, 0]))

    def test_processor_all_nans(self):
        """Limit test case when the image is all NaN values"""
        processor = BinarizationProcessor(method="otsu")
        nan_image = np.full((2, 2), np.nan)
        
        result = processor.transform_image(nan_image, self.mock_data)
        
        # If no valid values, give the original image back inaltered
        self.assertTrue(np.isnan(result).all())

    # Graphical interface test 

    def test_editor_ui_updates(self):
        """Tests if the graphical interface correctly loads when parameters are changed"""
        editor = BinarizationEditor()
        
        
        params = {
            "method": "kmeans",
            "manual_threshold": 0.7,
            "k_clusters": 4
        }
        editor.setParameters(params)
        
        
        self.assertEqual(editor.method, "kmeans")
        self.assertEqual(editor.k_clusters, 4)
        
        # Verify if combobox shows the correct value
        self.assertEqual(editor.cb_method.currentText(), "kmeans")
        
       
       
        self.assertFalse(editor.threshold_edit.isEnabled())
        self.assertTrue(editor.clusters_edit.isEnabled())

    def test_editor_createinstance(self):
        """Tests instance creation"""
        params = {
            "method": "otsu",
            "manual_threshold": 0.3,
            "k_clusters": 2
        }
        processor = BinarizationEditor.createinstance(params) 
        
        # Verifies if object is the correct selected method
        self.assertIsInstance(processor, BinarizationProcessor)
        self.assertEqual(processor.method, "otsu") 

if __name__ == "__main__":
    unittest.main()