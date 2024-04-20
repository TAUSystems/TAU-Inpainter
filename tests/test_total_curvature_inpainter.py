import unittest
from tau_inpainter.algorithms.total_curvature import TotalCurvatureInpainter

class TestTotalCurvatureInpainter(unittest.TestCase):
    def setUp(self):
        self.inpainter = TotalCurvatureInpainter()

    def test_instance(self):
        self.assertIsInstance(self.inpainter, TotalCurvatureInpainter)

if __name__ == '__main__':
    unittest.main()
