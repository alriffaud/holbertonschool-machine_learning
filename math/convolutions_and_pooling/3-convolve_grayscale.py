#!/usr/bin/env python3
""" This module defines the convolve_grayscale function. """
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    This function performs convolution on grayscale images with custom padding.
    Args:
        images (np.ndarray): a matrix of shape (m, h, w) containing multiple
        grayscale images. m is the number of images, h is the height in pixels
        and w is the width in pixels.
        kernel (np.ndarray): a matrix of shape (kh, kw) containing the kernel
        for the convolution. kh is the height of the kernel and kw is the width
        of the kernel.
        padding (tuple): a tuple of (Ph, Pw) containing the padding to apply.
        Ph is the padding for the height and Pw is the padding for the width.
        If padding is 'same', the output size is the same as the input size.
        If padding is 'valid', the output size is the input size minus the
        kernel size plus one.
        stride (tuple): a tuple of (Sh, Sw) containing the stride to apply. Sh
        is the stride for the height and Sw is the stride for the width.
    Returns: a np.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    if padding == 'valid':
        Ph = 0
        Pw = 0
    elif padding == 'same':
        Ph = kh // 2
        Pw = kw // 2
    else:
        Ph, Pw = padding
    padded_images = np.pad(images, ((0, 0), (Ph, Ph), (Pw, Pw)),
                           mode='constant')
    Sh, Sw = stride
    output_h = (h + 2 * Ph - kh) // Sh + 1
    output_w = (w + 2 * Pw - kw) // Sw + 1
    output = np.zeros((m, output_h, output_w))
    for i in range(output_h):
        for j in range(output_w):
            min_r = Sh * i
            max_r = min_r + kh
            min_c = Sw * j
            max_c = min_c + kw
            output[:, i, j] = np.sum(
                padded_images[:, min_r:max_r, min_c:max_c] * kernel,
                axis=(1, 2))
    return output
