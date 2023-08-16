import numpy as np
import scipy.ndimage as ndimage
import src.ImProUtils.filters as filters
import cv2


def canny(img, low_threshold, high_threshold, kernel_size, sigma=1):
    """Returns the canny edge detector of the image.
    :param img: the image to detect edges on (matrix form)
    :param low_threshold: the low threshold
    :param high_threshold: the high threshold
    :param kernel_size: size of the gaussian kernel
    :param sigma: the sigma value for the gaussian blur
    :return: the edges detected in the image
    """

    # Blur the image
    blurred_img = ndimage.convolve(img, filters.gaussian_kernel2d(kernel_size, sigma))

    # x derivative and y derivative using sobel
    sobel_x = filters.sobel_x_derivative(blurred_img)
    sobel_y = filters.sobel_y_derivative(blurred_img)

    # gradient magnitude and direction matrix
    grad_mag_mat = filters.gradient_magnitude(sobel_x, sobel_y)
    grad_dir_mat = filters.gradient_direction(sobel_x, sobel_y)
    quantized_dir_mat = filters.direction_quantization(grad_dir_mat)

    # non-maximum suppression
    suppressed_matrix = filters.non_maximum_suppression(grad_mag_mat, quantized_dir_mat)

    # hysteresis thresholding
    edges = filters.hysteresis_thresholding(suppressed_matrix, low_threshold, high_threshold)

    return edges


def harris():
    pass


def sift():
    pass