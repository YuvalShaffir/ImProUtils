"""
ImProUtils.filters module
=========================

Description:
------------
Useful functions for implementing filters on images.

Author:
-------
Yuval Shaffir

Date:
-----
15/08/2023
"""


import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal
from skimage import color


SAME_SHAPE_ERR = 'Arguments must have the same shape'
GRAYSCALE_ERR = 'Image must be grayscale'
LOW_IS_BIGGER_THAN_HIGH_ERR = 'low threshold must be smaller than high threshold'
POSITIVE_THRESHOLD_ERR = 'thresholds must be positive'
POSITIVE_KERNEL_ERR = 'sigma must be positive'
SIZE_MUST_BE_ODD_ERR = 'kernel_size must be odd'


def gaussian_kernel1d(kernel_size, sigma):
    """Returns a 1D gaussian kernel of size kernel_size and sigma."""
    base = np.linspace(-1, 1, kernel_size)
    base_kernel = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * (base / sigma) ** 2)
    return base_kernel


def gaussian_kernel2d(kernel_size, sigma):
    """Returns a 2D gaussian kernel of size kernel_size and sigma.
    I will use convolution because it is faster than the other methods."""
    # validate:
    if kernel_size % 2 == 0 or kernel_size < 0:
        raise ValueError(SIZE_MUST_BE_ODD_ERR)
    if sigma <= 0:
        raise ValueError(POSITIVE_KERNEL_ERR)

    # generate 1D kernel:
    base_kernel = gaussian_kernel1d(kernel_size, sigma)

    # generate 2D kernel:
    kernel = np.outer(base_kernel, base_kernel)
    kernel /= kernel.sum()
    return kernel


def sobel_x_derivative(img):
    """Returns the x derivative of the image using the 3x3 sobel operator."""
    # grayscale the image
    if len(img.shape) == 3:
        img = color.rgb2gray(img)

    sobel_x_kernel = np.outer(np.array([1, 2, 1]), np.array([1, 0, -1]))
    sobel_x = signal.convolve(sobel_x_kernel, img, mode='full').astype(np.float16)
    return sobel_x


def sobel_y_derivative(img):
    """Returns the y derivative of the image using the 3x3 sobel operator."""
    # grayscale the image
    if len(img.shape) == 3:
        img = color.rgb2gray(img)

    sobel_y_kernel = np.outer(np.array([1, 2, 1]), np.array([1, 0, -1])).T
    sobel_y = signal.convolve(sobel_y_kernel, img, mode='full')
    return sobel_y


def gradient_magnitude(grad_x, grad_y):
    """Returns the gradient magnitude of the image using the x and y derivatives."""
    # validate input:
    if grad_x.shape != grad_y.shape:
        raise ValueError(SAME_SHAPE_ERR)

    return np.sqrt(grad_x ** 2 + grad_y ** 2)


def gradient_direction(grad_x, grad_y):
    """Returns the gradient direction of the image using the x and y derivatives."""
    if grad_x.shape != grad_y.shape:
        raise ValueError(SAME_SHAPE_ERR)

    res = np.arctan2(grad_y, grad_x).astype(np.float16) * 180 / np.pi
    res[np.where(res < 0)] += 180
    return res


def direction_quantization(directions):
    """Returns the quantized direction of the image using the gradient direction."""
    inner_bins = np.array([0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180])
    bins = np.array([0, 45, 90, 135, 180])
    # for each value in the directions array, find the bin it belongs to (index in the bins array)
    inner_bin_indices = np.digitize(directions, inner_bins, right=False)

    bin_indices = np.digitize(inner_bins[inner_bin_indices], bins, right=False)

    # for each cell in the membership matrix (i, j) set the value to the bin value
    # The -1 is because the digitize returns out of bounds indices
    return bins[bin_indices-1]


def non_maximum_suppression(grad_matrix, phase_matrix):
    # validate arguments:
    if grad_matrix.shape != phase_matrix.shape:
        raise ValueError(SAME_SHAPE_ERR)

    suppressed_matrix = np.zeros(grad_matrix.shape)

    for i in range(grad_matrix.shape[0]):
        for j in range(grad_matrix.shape[1]):
            current_value = grad_matrix[i][j]
            angle = phase_matrix[i][j]
            neighbors = []
            if angle == 0:
                # 0 degrees - horizontal edge
                neighbors = grad_matrix[i - 1: i + 1][j]

            elif angle == 45:
                # 45 degrees - diagonal edge
                neighbors = [grad_matrix[i+1][j-1], grad_matrix[i-1][j+1]]

            elif angle == 90:
                # 90 degrees - vertical edge
                neighbors = grad_matrix[i][j - 1: j + 1]

            elif angle == 135 :
                # 135 degrees - diagonal edge
                neighbors = [grad_matrix[i - 1][j - 1], grad_matrix[i + 1][j + 1]]

            if current_value >= np.max(neighbors):
                # zero out the neighbors
                suppressed_matrix[i][j] = current_value

    return suppressed_matrix


def hysteresis_thresholding(suppressed_matrix, low_val, high_val):
    """Returns the hysteresis thresholding of the image using the low and high thresholds.
    :param suppressed_matrix: the matrix after non-maximum suppression
    :param low_val: the low threshold
    :param high_val: the high threshold
    :return: the edges of the image
    """
    # validate arguments:
    if low_val < 0 or high_val < 0:
        raise ValueError(POSITIVE_THRESHOLD_ERR)
    if low_val > high_val:
        raise ValueError(LOW_IS_BIGGER_THAN_HIGH_ERR)

    strong_edges = (suppressed_matrix >= high_val)
    weak_edges = (suppressed_matrix >= low_val) & (suppressed_matrix < high_val)

    # find weak edges that are connected to strong edges
    edges = np.zeros(suppressed_matrix.shape)
    edges[strong_edges] = 1
    for i in range(1, edges.shape[0] - 1):
        for j in range(1, edges.shape[1] - 1):
            # check if the current pixel is a weak edge and if it is connected to a strong edge
            # todo: fix the error here
            if weak_edges[i][j] and np.any(strong_edges[i - 1:i + 1, j - 1:j + 1]):
                edges[i][j] = 1

    return edges


def laplacian(img):
    """
    convolve the image with a laplacian kernel
    :param img: a grayscale image
    :return: the image after the convolution with the laplacian kernel
    """
    # grayscale the image
    if len(img.shape) == 3:
        img = color.rgb2gray(img)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return ndimage.convolve(img, kernel)
