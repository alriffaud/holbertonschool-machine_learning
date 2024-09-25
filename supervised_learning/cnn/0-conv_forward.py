#!/usr/bin/env python3
""" This module defines the conv_forward function. """
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    This function performs forward propagation over a convolutional layer
    of a neural network.
    Args:
        A_prev (np.ndarray): is a numpy array of shape (m, h_prev, w_prev,
            c_prev) containing the output of the previous layer.
            - m is the number of examples.
            - h_prev is the height of the previous layer.
            - w_prev is the width of the previous layer.
            - c_prev is the number of channels in the previous layer.
        W (np.ndarray): is a numpy array of shape (kh, kw, c_prev, c_new)
            containing the kernels for the convolution.
            - kh is the filter height.
            - kw is the filter width.
            - c_prev is the number of channels in the previous layer.
            - c_new is the number of channels in the output.
        b (np.ndarray): is a numpy array of shape (1, 1, 1, c_new) containing
            the biases applied to the convolution.
        activation (function): is the activation function applied to the
            convolution.
        padding (str, optional): is either a string that is 'same' or 'valid'.
            Defaults to 'same'.
        stride (tuple, optional): is a tuple of (sh, sw) containing the strides
            for the convolution.
            - sh is the stride for the height.
            - sw is the stride for the width.
            Defaults to (1, 1).
    Returns: The output of the convolutional layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    Sh, Sw = stride
    if padding == 'valid':
        Ph, Pw = 0, 0
    elif padding == 'same':
        Ph = ((h_prev - 1) * Sh + kh - h_prev) // 2
        Pw = ((w_prev - 1) * Sw + kw - w_prev) // 2
    else:
        Ph, Pw = padding
    padded_images = np.pad(A_prev, ((0, 0), (Ph, Ph), (Pw, Pw), (0, 0)),
                           mode='constant')
    output_h = (h_prev + 2 * Ph - kh) // Sh + 1
    output_w = (w_prev + 2 * Pw - kw) // Sw + 1
    output = np.zeros((m, output_h, output_w, c_new))
    # Reshape kernels for convolution
    W = W.reshape((1, *W.shape))
    for i in range(output_h):
        for j in range(output_w):
            min_r = Sh * i
            max_r = min_r + kh
            min_c = Sw * j
            max_c = min_c + kw
            output[:, i, j, :] = np.sum(
                padded_images[:, min_r:max_r, min_c:max_c, :, None] * W,
                axis=(1, 2, 3)) + b
    output = activation(output)
    return output
