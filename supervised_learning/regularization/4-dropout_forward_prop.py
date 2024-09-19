#!/usr/bin/env python3
""" This module defines the dropout_forward_prop function. """
import numpy as np


def softmax_activation(Z):
    """
    This function applies the softmax activation function.
    """
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    This function conducts forward propagation using dropout.
    All layers except the last use the tanh activation function.
    The last layer uses the softmax activation function.
    Args:
        X (tf.Tensor): is a tensor of shape (nx, m) containing the input data
            for the network. nx is the number of input features to the network,
            and m is the number of data points.
        weights (dict): is a dictionary of the weights and biases of the neural
            network.
        L (int): is the number of layers in the network.
        keep_prob (float): is the probability that a node will be kept.
    Returns: a dictionary containing the outputs of each layer and the dropout
        mask used on each layer.
    """
    cache = {'A0': X}

    for i in range(1, L + 1):
        # Calculate the pre-activation linear combination of weights and inputs
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]
        Z = np.matmul(W, A_prev) + b
        if i != L:
            # Apply tanh activation function and dropout to hidden layers
            A = np.tanh(Z)
            # Apply the Dropout mask to the hidden layers
            random_array = np.random.rand(A.shape[0], A.shape[1])
            D = (random_array < keep_prob).astype(int)
            A = np.multiply(A, D)
            A /= keep_prob
            cache["D" + str(i)] = D
        else:
            # Apply softmax activation function to the output layer
            A = softmax_activation(Z)
        cache["A" + str(i)] = A

    return cache
