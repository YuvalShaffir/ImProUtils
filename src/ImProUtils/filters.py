import numpy as np
from PIL import Image
import scipy.ndimage as ndimage


def gaussian_kernel1d(kernel_size, sigma):
    base = np.linspace(-1, 1, kernel_size)
    base_kernel = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * (base / sigma) ** 2)
    return base_kernel


def gaussian_kernel2d(kernel_size, sigma):
    """Returns a gaussian kernel of size kernel_size and sigma.
    I will use convolution because it is faster than the other methods."""
    # validate:
    if kernel_size % 2 == 0 or kernel_size < 0:
        raise ValueError('kernel_size must be odd')
    if sigma <= 0:
        raise ValueError('sigma must be positive')

    # generate 1D kernel:
    base_kernel = gaussian_kernel1d(kernel_size, sigma)

    # generate 2D kernel:
    kernel = np.outer(base_kernel, base_kernel)
    return kernel




def sobel_x_derivative():
    pass


def sobel_y_derivative():
    pass


def gradient_magnitude():
    pass


def non_maximum_suppression():
    pass


def hysteresis_thresholding():
    pass


def histogram_eq():
    pass


def laplacian():
    pass

