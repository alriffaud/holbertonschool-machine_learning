#!/usr/bin/env python3
""" This module defines the convolve_grayscale_same function. """
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    This function performs a same convolution on grayscale images.
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
    Ph = kh // 2
    Pw = kw // 2
    padded_images = np.pad(images, ((0, 0), (Ph, Ph), (Pw, Pw)),
                           mode='constant')
    output = np.zeros((m, h, w))
    for i in range(h):
        for j in range(w):
            output[:, i, j] = np.sum(
                padded_images[:, i: i + kh, j: j + kw] * kernel,
                axis=(1, 2))
    return output
