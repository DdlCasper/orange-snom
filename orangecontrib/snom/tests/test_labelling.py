import unittest
import numpy as np
from AnyQt.QtWidgets import QApplication

from orangecontrib.snom.labelling import InstanceLabelingProcessor, InstanceLabelingEditor

app = QApplication.instance() or QApplication([])

class TestInstanceLabelingProcessor(unittest.TestCase):
    def setUp(self):
        self.mock_data = None
        # Image with 2 blobs, one on the left and the other on the right
        self.test_image = np.array([
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0]
        ])

    def test_connected_components(self):
        processor = InstanceLabelingProcessor(method="connected_components")
        result = processor.transform_image(self.test_image, self.mock_data)
        
        # Verify that beckground stays 0 
        self.assertEqual(result[0, 3], 0.0)
        # Verify that the 2 blobs have different labels (>0)
        label_1 = result[0, 0]
        label_2 = result[3, 3]
        self.assertNotEqual(label_1, label_2)
        self.assertGreater(label_1, 0)
        self.assertGreater(label_2, 0)

class TestInstanceLabelingEditor(unittest.TestCase):
    def setUp(self):
        self.editor = InstanceLabelingEditor(None)

    def test_ui_logic(self):
        # The input distance_edit must be active only for watershed
        self.editor.setParameters({"method": "connected_components"})
        self.assertFalse(self.editor.distance_edit.isEnabled())
        
        self.editor.setParameters({"method": "watershed"})
        self.assertTrue(self.editor.distance_edit.isEnabled())

if __name__ == "__main__":
    unittest.main()