#!/usr/bin/env python3
""" This module defines the conv_backward function. """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    This function performs back propagation over a convolutional layer of a
    neural network.
    Args:
        dZ (np.ndarray): is a numpy array of shape (m, h_new, w_new, c_new)
            containing the partial derivatives with respect to the unactivated
            output of the convolutional layer.
            - m is the number of examples.
            - h_new is the height of the output.
            - w_new is the width of the output.
            - c_new is the number of channels in the output.
        A_prev (np.ndarray): is a numpy array of shape (m, h_prev, w_prev,
            c_prev) containing the output of the previous layer.
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
        padding (str, optional): is either a string that is 'same' or 'valid'.
            Defaults to 'same'.
        stride (tuple, optional): is a tuple of (sh, sw) containing the strides
            for the convolution.
            - sh is the stride for the height.
            - sw is the stride for the width.
            Defaults to (1, 1).
    Returns: The partial derivatives with respect to the previous layer
        (dA_prev), the kernels (dW), and the biases (db), respectively.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    Sh, Sw = stride
    m, h_new, w_new, c_new = dZ.shape
    if padding == 'valid':
        Ph, Pw = 0, 0
    elif padding == 'same':
        Ph = ((h_prev - 1) * Sh + kh - h_prev) // 2 + 1
        Pw = ((w_prev - 1) * Sw + kw - w_prev) // 2 + 1
    else:
        Ph, Pw = padding
    padded_images = np.pad(A_prev, ((0, 0), (Ph, Ph), (Pw, Pw), (0, 0)),
                           mode='constant')
    dA_prev = np.zeros_like(padded_images)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    for i in range(h_new):
        for j in range(w_new):
            min_r = Sh * i
            max_r = min_r + kh
            min_c = Sw * j
            max_c = min_c + kw
            for k in range(c_new):
                dA_prev[:, min_r:max_r, min_c:max_c, :] += (
                    W[:, :, :, k] * dZ[:, i, j, k, np.newaxis, np.newaxis,
                                       np.newaxis])
                dW[:, :, :, k] += np.sum(
                    padded_images[:, min_r:max_r, min_c:max_c, :] *
                    dZ[:, i, j, k, np.newaxis, np.newaxis, np.newaxis], axis=0)
    if padding == 'same':
        dA_prev = dA_prev[:, Ph:-Ph, Pw:-Pw, :]
    return dA_prev, dW, db
