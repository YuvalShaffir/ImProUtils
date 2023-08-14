import unittest
import os
import numpy as np
from PIL import Image
import src.ImProUtils.image as ImProImage

# Constants:

# Test 1
legal_file_path1 = os.path.abspath('test_images\\pizza_pixel_art.jpg')
legal_mode1 = 'RGB'

# Test 3
illegal_file_type = os.path.abspath('test_images\\pizza_recipes.pdf')


def test_image_from_legal_file():
    """Test 1: image_from_file() with legal arguments."""
    try:
        # RGB image:
        img = ImProImage.image_from_file(legal_file_path1, 'RGB')
        assert isinstance(img, np.ndarray)
        assert img.shape[2] == 3
        assert img.shape[0] == 1080
        assert img.shape[1] == 1000
        assert img.dtype == np.uint8

        # grayscale image:
        img = ImProImage.image_from_file(legal_file_path1, 'L')
        assert isinstance(img, np.ndarray)
        assert img.shape[0] == 1080
        assert img.shape[1] == 1000
        assert img.dtype == np.uint8
    except Exception as e:
        assert False, f'Unexpected error: {e}'


def test_image_from_illegal_file():
    try:
        img = ImProImage.image_from_file('illegal_file_path', 'RGB')
    except FileNotFoundError:
        pass


def test_image_from_illegal_type():
    try:
        img = ImProImage.image_from_file(illegal_file_type, 'RGB')
    except FileNotFoundError:
        pass


def test_image_from_illegal_mode():
    try:
        img = ImProImage.image_from_file(legal_file_path1, 'illegal_mode')
    except ValueError:
        pass


if __name__ == '__main__':
    test_image_from_legal_file()
    test_image_from_illegal_file()
    test_image_from_illegal_type()
    test_image_from_illegal_mode()