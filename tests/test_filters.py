import unittest
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
import src.ImProUtils.filters as ImProFilters


class TestGaussianFilter(unittest.TestCase):
    def gaussian_filter_check(self, matrix):
        self.assertTrue(np.all(matrix >= 0))
        self.assertAlmostEqual(np.sum(matrix), 1)

    def test_normal_values(self):
        # test 1: normal case
        res1 = ImProFilters.gaussian_kernel2d(3, 1)
        # check if all values are positive and sum to 1:
        self.gaussian_filter_check(res1)
        print("\n")
        print(f"Kernel size: 3, Sigma = 1:\n {res1} \n")

        # test 2: big kernel size
        res2 = ImProFilters.gaussian_kernel2d(17, 5)
        # check if all values are positive and sum to 1:
        self.gaussian_filter_check(res2)
        print(f"Kernel size: 17, Sigma = 5:\n {res2} \n")

        # test 3: big sigma
        res3 = ImProFilters.gaussian_kernel2d(3, 50)
        # check if all values are positive and sum to 1:
        self.gaussian_filter_check(res3)
        print(f"Kernel size: 3, Sigma = 50:\n {res3} \n")

    def test_negative_sigma(self):
        with self.assertRaises(ValueError):
            ImProFilters.gaussian_kernel2d(3, -1)

    def test_even_kernel_size(self):
        with self.assertRaises(ValueError):
            ImProFilters.gaussian_kernel2d(4, 1)

    def test_zero_kernel_size(self):
        with self.assertRaises(ValueError):
            ImProFilters.gaussian_kernel2d(0, 1)

    def test_negative_kernel_size(self):
        with self.assertRaises(ValueError):
            ImProFilters.gaussian_kernel2d(-1, 1)


class TestSobel(unittest.TestCase):
    def test_correct_sobel_range(self):
        res_x = ImProFilters.sobel_x_derivative(Image.open('test_images/pizza_pixel_art.jpg'))
        res_y = ImProFilters.sobel_y_derivative(Image.open('test_images/pizza_pixel_art.jpg'))
        # check if all values are in range [-255, 255]:
        self.assertTrue(np.all(res_x >= -255) and np.all(res_x <= 255))
        self.assertTrue(np.all(res_y >= -255) and np.all(res_y <= 255))

    def test_sobel_y_derivative(self):
        pass


