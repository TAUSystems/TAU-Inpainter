import unittest

import numpy as np

from tau_inpainter.algorithms.total_curvature import TotalCurvatureInpainter

class TestTotalCurvatureInpainter(unittest.TestCase):
    def setUp(self):
        self.inpainter = TotalCurvatureInpainter()

    def test_instance(self):
        self.assertIsInstance(self.inpainter, TotalCurvatureInpainter)

    def test_inpaint(self):
        image = np.zeros((5, 5), dtype=float)
        image[2, 2] = 1.0
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        inpainted = self.inpainter.inpaint(image, mask)
        np.testing.assert_allclose(inpainted, 0.0, atol=1e-6)

if __name__ == '__main__':
    unittest.main()
