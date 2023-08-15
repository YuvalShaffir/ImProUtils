import unittest
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
import src.ImProUtils.filters as ImProFilters


class TestGaussianFilter(unittest.TestCase):
    def test_gaussian_filter(self):
        print(f"2D Kernel:\n {ImProFilters.gaussian_kernel2d(3, 1)}")
