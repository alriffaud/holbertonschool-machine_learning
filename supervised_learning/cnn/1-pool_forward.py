#!/usr/bin/env python3
""" This module defines the pool_forward function. """
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    This function performs forward propagation over a pooling layer of a neural
    network.
    Args:
        A_prev (np.ndarray): is a numpy array of shape (m, h_prev, w_prev,
            c_prev) containing the output of the previous layer.
            - m is the number of examples.
            - h_prev is the height of the previous layer.
            - w_prev is the width of the previous layer.
            - c_prev is the number of channels in the previous layer.
        kerne_shape (np.ndarray): a matrix of shape (kh, kw) containing the
            kernel shape for the pooling.
            - kh is the height of the kernel.
            - kw is the width of the kernel.
        stride (tuple): a tuple of (Sh, Sw) containing the strides for the
            pooling.
            - Sh is the stride for the height.
            - Sw is the stride for the width.
        mode (str): a string containing the type of pooling. 'max' indicates
            max pooling and 'avg' indicates average pooling.
    Returns: the output of the pooling layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    Sh, Sw = stride
    output_h = (h_prev - kh) // Sh + 1
    output_w = (w_prev - kw) // Sw + 1
    output = np.zeros((m, output_h, output_w, c_prev))
    for i in range(output_h):
        for j in range(output_w):
            min_r = Sh * i
            max_r = min_r + kh
            min_c = Sw * j
            max_c = min_c + kw
            if mode == 'max':
                output[:, i, j, :] = np.max(
                    A_prev[:, min_r:max_r, min_c:max_c, :],
                    axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(
                    A_prev[:, min_r:max_r, min_c:max_c, :],
                    axis=(1, 2))
    return output
