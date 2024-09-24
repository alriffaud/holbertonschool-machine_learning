#!/usr/bin/env python3
""" This module defines the pool function. """
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    This function performs pooling on images.
    Args:
        images (np.ndarray): a matrix of shape (m, h, w, c) containing multiple
        images. m is the number of images, h is the height in pixels, w is the
        width in pixels and c is the number of channels in the image.
        kerne_shape (np.ndarray): a matrix of shape (kh, kw) containing the
        kernel shape for the pooling. kh is the height of the kernel and kw is
        the width of the kernel.
        stride (tuple): a tuple of (Sh, Sw) containing the stride to apply. Sh
        is the stride for the height and Sw is the stride for the width.
        mode (str): a string containing the type of pooling. 'max' indicates
        max pooling and 'avg' indicates average pooling.
    Returns: a np.ndarray containing the convolved images.
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    Sh, Sw = stride
    output_h = (h - kh) // Sh + 1
    output_w = (w - kw) // Sw + 1
    output = np.zeros((m, output_h, output_w, c))
    for i in range(output_h):
        for j in range(output_w):
            min_r = Sh * i
            max_r = min_r + kh
            min_c = Sw * j
            max_c = min_c + kw
            if mode == 'max':
                output[:, i, j, :] = np.max(
                    images[:, min_r:max_r, min_c:max_c, :],
                    axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(
                    images[:, min_r:max_r, min_c:max_c, :],
                    axis=(1, 2))
    return output
