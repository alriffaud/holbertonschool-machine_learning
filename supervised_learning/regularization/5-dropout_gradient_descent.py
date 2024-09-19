#!/usr/bin/env python3
""" This module defines the dropout_gradient_descent function. """
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    This function updates the weights of a neural network with Dropout
    regularization using gradient descent. The neural network uses tanh
    activations on each layer except the last, which uses a softmax activation.
    The weights of the network should be updated in place
    Args:
        Y (numpy.ndarray): is a one-hot numpy.ndarray of shape (classes, m)
            that contains the correct labels for the data. classes is the
            number of classes, and m is the number of data points.
        weights (dict): is a dictionary of the weights and biases of the
        neural network.
        cache (dict): is a dictionary of the outputs of each layer of the
        neural network.
        alpha (float): is the learning rate.
        keep_prob (float): is the probability that a node will be kept.
        L (int): is the number of layers of the network.
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        if i > 1:
            D = cache["D" + str(i - 1)]
            dA_prev = np.matmul(weights["W" + str(i)].T, dZ)
            dA_prev = dA_prev * D
            dA_prev = dA_prev / keep_prob
            dZ = dA_prev * (1 - np.power(A_prev, 2))
        weights['W' + str(i)] = W - alpha * dW
        weights['b' + str(i)] = b - alpha * db
