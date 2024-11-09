#!/usr/bin/env python3
""" This module defines the convolve function. """
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    This function performs convolution on images using multiple kernels.
    Args:
        images (np.ndarray): a matrix of shape (m, h, w, c) containing multiple
            images. m is the number of images, h is the height in pixels, w is
            the width in pixels and c is the number of channels in the image.
        kernels (np.ndarray): a matrix of shape (kh, kw, c, nc) containing the
            kernel for the convolution. kh is the height of the kernel, kw is
            the width of the kernel and nc is the number of kernels.
        padding (tuple): a tuple of (Ph, Pw) containing the padding to apply.
            Ph is the padding for the height and Pw is the padding for the
            width. If padding is 'same', the output size is the same as the input
            size. If padding is 'valid', the output size is the input size minus
            the kernel size plus one.
        stride (tuple): a tuple of (Sh, Sw) containing the stride to apply. Sh
            is the stride for the height and Sw is the stride for the width.
    Returns: a np.ndarray containing the convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    Sh, Sw = stride
    if padding == 'valid':
        Ph, Pw = 0, 0
    elif padding == 'same':
        Ph = ((h - 1) * Sh + kh - h) // 2 + 1
        Pw = ((w - 1) * Sw + kw - w) // 2 + 1
    else:
        Ph, Pw = padding
    padded_images = np.pad(images, ((0, 0), (Ph, Ph), (Pw, Pw), (0, 0)),
                           mode='constant')
    output_h = (h + 2 * Ph - kh) // Sh + 1
    output_w = (w + 2 * Pw - kw) // Sw + 1
    output = np.zeros((m, output_h, output_w, nc))
    # Reshape kernels for convolution
    kernels = kernels.reshape((1, *kernels.shape))
    for i in range(output_h):
        for j in range(output_w):
            min_r = Sh * i
            max_r = min_r + kh
            min_c = Sw * j
            max_c = min_c + kw
            output[:, i, j, :] = np.sum(
                padded_images[:, min_r:max_r, min_c:max_c, :, None] * kernels,
                axis=(1, 2, 3))
    return output
