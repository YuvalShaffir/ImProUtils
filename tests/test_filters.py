import unittest
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
import src.ImProUtils.filters as ImProFilters
import src.ImProUtils.image as ImProImage
import cv2
from matplotlib import pyplot as plt


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
        """Tests if the sobel derivative is in the range [-255, 255]"""
        res_x = ImProFilters.sobel_x_derivative(ImProImage.image_from_file('test_images/pizza_pixel_art.jpg'))
        res_y = ImProFilters.sobel_y_derivative(ImProImage.image_from_file('test_images/pizza_pixel_art.jpg'))
        # check if all values are in range [-255, 255]:
        self.assertTrue(np.all(res_x >= -255) and np.all(res_x <= 255))
        self.assertTrue(np.all(res_y >= -255) and np.all(res_y <= 255))

    def test_sobel_x_derivative(self):
        """Tests if the sobel derivative is correct"""
        my_res = ImProFilters.sobel_x_derivative(ImProImage.image_from_file('test_images/pizza_pixel_art.jpg'))
        scipy_res = ndimage.sobel(ImProImage.image_from_file('test_images/pizza_pixel_art.jpg'), axis=1)
        cv2_res = cv2.Sobel(np.array(ImProImage.image_from_file('test_images/pizza_pixel_art.jpg')), cv2.CV_16SC1, 1, 0, ksize=3)
        print(f"my_res: {my_res}\n")
        plt.imshow(my_res, cmap='gray')
        plt.title("my_res")
        plt.show()

        print(f"\nscipy_res: {scipy_res}\n")
        plt.imshow(scipy_res, cmap='gray')
        plt.title("scipy_res:")
        plt.show()

        print(f"\ncv2_res: {cv2_res}, tpye: {cv2_res.dtype}\n")
        plt.imshow(cv2_res, cmap='gray')
        plt.title("cv2_res:")
        plt.show()

    def test_sobel_y_derivative(self):
        """Tests if the sobel derivative is correct"""
        my_res = ImProFilters.sobel_y_derivative(ImProImage.image_from_file('test_images/pizza_pixel_art.jpg'))
        scipy_res = ndimage.sobel(ImProImage.image_from_file('test_images/pizza_pixel_art.jpg'), axis=0)
        cv2_res = cv2.Sobel(np.array(ImProImage.image_from_file('test_images/pizza_pixel_art.jpg')), cv2.CV_16SC1, 0, 1, ksize=3)

        print(f"my_res: {my_res}\n")
        plt.imshow(my_res, cmap='gray')
        plt.title("my_res")
        plt.show()

        print(f"\nscipy_res: {scipy_res}\n")
        plt.imshow(scipy_res, cmap='gray')
        plt.title("scipy_res:")
        plt.show()

        print(f"\ncv2_res: {cv2_res}, tpye: {cv2_res.dtype}\n")
        plt.imshow(cv2_res, cmap='gray')
        plt.title("cv2_res:")
        plt.show()


class TestGradient(unittest.TestCase):
    def test_gradient_magnitude(self):
        """Tests if the gradient magnitude is correct"""
        pass


class TestNonMaxSup(unittest.TestCase):
    pass

class TestHysterisis(unittest.TestCase):
    pass