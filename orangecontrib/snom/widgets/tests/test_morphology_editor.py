import unittest
import numpy as np
from AnyQt.QtWidgets import QApplication

from  orangecontrib.snom.widgets.preprocessors.morphology_editor import MorphologyProcessor, MorphologyEditor

app = QApplication.instance() or QApplication([])

class TestMorphologyProcessor(unittest.TestCase):
    def setUp(self):
        # We create a 3x3 square of 1.0 sorrounded by 0.0 in a 5xd5 grid
        self.test_image = np.zeros((5, 5))
        self.test_image[1:4, 1:4] = 1.0
        self.mock_data = None

    def test_erosion(self):
        # Erosion with 3x3 kernes should reduce the 3x3 square in a single pixel
        processor = MorphologyProcessor(method="erosion", size=3)
        result = processor.transform_image(self.test_image, self.mock_data)
        
        self.assertEqual(result[2, 2], 1.0) # Centre remains 1
        self.assertEqual(result[1, 1], 0.0) # Berders become 0

    def test_dilation(self):
        processor = MorphologyProcessor(method="dilation", size=3)
        result = processor.transform_image(self.test_image, self.mock_data)
        
        # Dilation should expand the 3x3 square untile the whole 5x5 grid is filled 
        self.assertEqual(result[0, 0], 1.0) 
        self.assertEqual(result[4, 4], 1.0)

class TestMorphologyEditor(unittest.TestCase):
    def test_createinstance(self):
        params = {"method": "closing", "size": 5}
        processor = MorphologyEditor.createinstance(params)
        self.assertEqual(processor.method, "closing")[cite: 4]
        self.assertEqual(processor.size, 5)[cite: 4]

if __name__ == "__main__":
    unittest.main()