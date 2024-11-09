#!/usr/bin/env python3
""" This module defines the convolve_grayscale_padding function. """
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    This function performs convolution on grayscale images with custom padding.
    Args:
        images (np.ndarray): a matrix of shape (m, h, w) containing multiple
            grayscale images. m is the number of images, h is the height in
            pixels and w is the width in pixels.
        kernel (np.ndarray): a matrix of shape (kh, kw) containing the kernel
            for the convolution. kh is the height of the kernel and kw is the
            width of the kernel.
        padding (tuple): a tuple of (Ph, Pw) containing the padding to apply.
        Ph is the padding for the height and Pw is the padding for the width.
    Returns: a np.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    Ph, Pw = padding
    padded_images = np.pad(images, ((0, 0), (Ph, Ph), (Pw, Pw)),
                           mode='constant')
    output_h = h + 2 * Ph - kh + 1
    output_w = w + 2 * Pw - kw + 1
    output = np.zeros((m, output_h, output_w))
    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(
                padded_images[:, i: i + kh, j: j + kw] * kernel,
                axis=(1, 2))
    return output
