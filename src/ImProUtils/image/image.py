"""
ImProUtils.image.image
======================

Description:
------------
Useful functions for simple image importation.

Author:
-------
Yuval Shaffir

Date:
-----
13/08/2023

Accepted file types:
-------------------
BLP, BMP, BUFR, CUR, DCX, DDS, DIB, EPS, FITS, FLI, FTEX, GBR, GIF, GRIB,
HDF5, ICNS, ICO, IM, IMT, IPTC, JPEG, JPEG2000, MCIDAS, MPEG, MSP, PCD, PCX,
PIXAR, PNG, PPM, PSD, QOI, SGI, SPIDER, SUN, TGA, TIFF, WEBP, WMF, XBM, XPM, XVTHUMB

Accepted modes:
---------------
1 (1-bit pixels, black and white, stored with one pixel per byte)
L (8-bit pixels, grayscale)
P (8-bit pixels, mapped to any other mode using a color palette)
RGB (3x8-bit pixels, true color)
RGBA (4x8-bit pixels, true color with transparency mask)
CMYK (4x8-bit pixels, color separation)
YCbCr (3x8-bit pixels, color video format)
Note that this refers to the JPEG, and not the ITU-R BT.2020, standard
LAB (3x8-bit pixels, the L*a*b color space)
HSV (3x8-bit pixels, Hue, Saturation, Value color space)
Hue’s range of 0-255 is a scaled version of 0 degrees <= Hue < 360 degrees
I (32-bit signed integer pixels)
F (32-bit floating point pixels)
"""

# Imports
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import urllib.request

SUCCESS_MSG = 'Importing Success!\n'
ARRAY_IS_NULL_ERR = 'Array is null!\n'
POSITIVE_VAL_ERR = 'Number of columns must be positive!\n'
MUST_BE_LIST_ERR = 'Array must be a list!\n'


def image_from_file(path, mode='RGB', err_raise=True, print_info=True):
    """
    Import an image from a file.
    Using PIL.Image to import the image, which is considered the fastest library for that.

    :param: path:  path to the image file. format must be one of the accepted formats (see description).
    :param: mode:  mode of the image (E.g. RGB, L, etc.. more info in the description).
    :param: err_raise: if True, raise an error if the file is not found.
    :param: print_info: if True, print info about the progress.
    :return: matrix of file in the given mode.
    """
    # check if file exists
    path = os.path.abspath(path)
    if not path:
        raise FileNotFoundError(f'Path: {path} not found!\n')

    with Image.open(path) as img:
        if print_info:
            print(SUCCESS_MSG)
        img = img.convert(mode)
        img = np.array(img)
        return img


def images_from_folder(path, mode='RGB', err_raise=True, print_info=True):
    """Get images from a folder.

    :param: path: path to folder.
    :param: mode: mode of images.
    :param: err_raise: if True, raise an error if the file is not found.
    :param: print_info: if True, print info about the progress.
    :return: list of images.
    """
    # check if folder exists
    path = os.path.abspath(path)
    if not path:
        raise FileNotFoundError(f'Path: {path} not found!\n')

    lst = np.array([])
    if print_info:
        print(f'Importing images from folder: {path}\n')

    for i in tqdm(os.listdir(path)):
        try:
            lst = np.append(lst, image_from_file(path + i, mode, err_raise, False))
        except FileNotFoundError:
            print(f"File not found! path: {path + i} \n")
    if print_info:
        print(SUCCESS_MSG)

    return lst


def img_from_1d_array(arr, cols):
    """
    Create an 2D matrix representing a grayscale image from a 1D array.

    :param: cols: number of columns in the image.
    :param: arr: 1D array of pixels.
    :return: matrix (image).
    """
    # validate input
    if not arr:
        raise ValueError(ARRAY_IS_NULL_ERR)
    if cols < 1:
        raise ValueError(POSITIVE_VAL_ERR)
    if not isinstance(arr, list):
        raise TypeError(MUST_BE_LIST_ERR)

    # reshape array into a matrix with the given number of columns
    return np.reshape(arr, (-1, cols))


def img_from_url(link, mode='RGB', err_raise=True, print_info=True):
    """
    Import an image from url.

    :param: link: link to image.
    :param: mode: mode of image.
    :param: err_raise: if True, raise an error if the file is not found.
    :param: print_info: if True, print info about the progress.
    :return: matrix of image.
    """
    # check if link exists
    if not link:
        raise FileNotFoundError(f'Link: {link} not found!\n')

    urllib.request.urlretrieve(link, 'temp.jpg')

    return image_from_file('temp.jpg', mode, err_raise, print_info)

