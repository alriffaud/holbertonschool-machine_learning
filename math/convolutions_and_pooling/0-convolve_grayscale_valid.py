#!/usr/bin/env python3
""" This module defines the convolve_grayscale_valid function. """
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    This function performs a valid convolution on grayscale images.
    Args:
        images (np.ndarray): a matrix of shape (m, h, w) containing multiple
            grayscale images. m is the number of images, h is the height in
            pixels and w is the width in pixels.
        kernel (np.ndarray): a matrix of shape (kh, kw) containing the kernel
            for the convolution. kh is the height of the kernel and kw is the
            width of the kernel.
    Returns: a np.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    output_h = h - kh + 1
    output_w = w - kw + 1
    output = np.zeros((m, output_h, output_w))
    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(images[:, i: i + kh, j: j + kw] * kernel,
                                     axis=(1, 2))
    return output
