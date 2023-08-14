"""
image_import_test.py
======================

Description:
------------
Tests the image.py module.

Author:
-------
Yuval Shaffir

Date:
-----
13/08/2023
"""

import unittest
import os
import numpy as np
from PIL import Image
import PIL
import src.ImProUtils.image as ImProImage

# Constants:
# Test 1
legal_file_path1 = os.path.abspath('test_images\\pizza_pixel_art.jpg')
legal_mode1 = 'RGB'
# Test 3
illegal_file_type = os.path.abspath('test_images\\pizza_recipes.pdf')


class TestFileImportation(unittest.TestCase):

    def test_image_from_legal_file(self):
        """Test 1: image_from_file() with legal arguments."""

        # RGB image:
        img = ImProImage.image_from_file(legal_file_path1, 'RGB')
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.shape[2], 3)
        self.assertEqual(img.shape[0], 1080)
        self.assertEqual(img.shape[1], 1000)
        self.assertEqual(img.dtype, np.uint8)

        # grayscale image:
        img = ImProImage.image_from_file(legal_file_path1, 'L')
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.shape[0], 1080)
        self.assertEqual(img.shape[1], 1000)
        self.assertEqual(img.dtype, np.uint8)

    def test_image_from_illegal_file(self):
        with self.assertRaises(IOError):
            ImProImage.image_from_file('illegal_file_path', 'RGB')

    def test_image_from_illegal_type(self):
        with self.assertRaises(PIL.UnidentifiedImageError):
            ImProImage.image_from_file(illegal_file_type, 'RGB')

    def test_image_from_illegal_mode(self):
        with self.assertRaises(ValueError):
            ImProImage.image_from_file(legal_file_path1, 'illegal_mode')


class TestFolderImport(unittest.TestCase):
    def test_legal_folder(self):
        lst = ImProImage.images_from_folder('legal_img_folder\\')
        self.assertIsInstance(lst, np.ndarray)
        # Check the number of images in the folder is equal to the number of images imported:
        self.assertEqual(len(lst), len([name for name in os.listdir('legal_img_folder\\') if os.path.isfile(name)]))
        # Check that all images are of type np.ndarray:
        for img in lst:
            self.assertIsInstance(img, np.ndarray)

    def test_illegal_folder_path(self):
        with self.assertRaises(FileNotFoundError):
            ImProImage.images_from_folder('illegal_folder_path\\')

    def test_illegal_file_inside_folder(self):
        with self.assertRaises(PIL.UnidentifiedImageError):
            ImProImage.images_from_folder('test_images\\')


class Test1DArrayToImage(unittest.TestCase):
    pass


class TestImageFromUrl(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
