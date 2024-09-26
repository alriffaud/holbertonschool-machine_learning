#!/usr/bin/env python3
""" This module defines the pool_backward function. """
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    This function performs back propagation over a pooling layer of a neural
    network.
    Args:
        dA (np.ndarray): is a numpy array of shape (m, h_new, w_new, c_new)
            containing the partial derivatives with respect to the output of
            the pooling layer.
            - m is the number of examples.
            - h_new is the height of the output.
            - w_new is the width of the output.
            - c_new is the number of channels in the output.
        A_prev (np.ndarray): is a numpy array of shape (m, h_prev, w_prev,
            c_prev) containing the output of the previous layer.
            - h_prev is the height of the previous layer.
            - w_prev is the width of the previous layer.
            - c_prev is the number of channels in the previous layer.
        kernel_shape (np.ndarray): a matrix of shape (kh, kw) containing the
            kernel shape for the pooling.
            - kh is the height of the kernel.
            - kw is the width of the kernel.
        stride (tuple): a tuple of (Sh, Sw) containing the strides for the
            pooling.
            - Sh is the stride for the height.
            - Sw is the stride for the width.
        mode (str): a string containing the type of pooling. 'max' indicates
            max pooling and 'avg' indicates average pooling.
    Returns: the partial derivatives with respect to the previous layer
        (dA_prev).
    """
    m, h_new, w_new, c = dA.shape
    h_prev, w_prev, c_prev = A_prev.shape[1:]
    kh, kw = kernel_shape
    Sh, Sw = stride
    dA_prev = np.zeros_like(A_prev)
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for f in range(c):
                    v_start = h * Sh
                    v_end = v_start + kh
                    h_start = w * Sw
                    h_end = h_start + kw
                    if mode == 'avg':
                        avg_dA = dA[i, h, w, f] / kh / kw
                        dA_prev[i, v_start:v_end, h_start:h_end, f] +=\
                            (np.ones((kh, kw)) * avg_dA)
                    elif mode == 'max':
                        region = A_prev[i, v_start:v_end, h_start:h_end, f]
                        mask = (region == np.max(region))
                        dA_prev[i, v_start:v_end, h_start:h_end, f] +=\
                            mask * dA[i, h, w, f]
    return dA_prev
