#!/usr/bin/env python3
""" This module defines the bi_rnn function. """
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    This function performs forward propagation for a bidirectional RNN
    Arguments:
      - bi_cell is an instance of BidirectionalCell that will be used for the
        forward propagation
      - X is a numpy.ndarray of shape (t, m, i) that contains the data to be
        used
        - t is the maximum number of time steps
        - m is the batch size
        - i is the dimensionality of the data
      - h_0 is a numpy.ndarray of shape (m, h) containing the initial hidden
        state
        - h is the dimensionality of the hidden state
      - h_t is a numpy.ndarray of shape (m, h) containing the terminal hidden
        state
    Returns: H, Y
      - H is a numpy.ndarray containing all of the hidden states
      - Y is a numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    h = h_0.shape[1]

    # Initialize hidden states for forward and backward directions
    H_forward = np.zeros((t, m, h))
    H_backward = np.zeros((t, m, h))

    # Forward direction
    h_f = h_0
    for step in range(t):
        h_f = bi_cell.forward(h_f, X[step])
        H_forward[step] = h_f

    # Backward direction
    h_b = h_t
    for step in reversed(range(t)):
        h_b = bi_cell.backward(h_b, X[step])
        H_backward[step] = h_b

    # Concatenate forward and backward hidden states
    H = np.concatenate((H_forward, H_backward), axis=2)

    # Compute the output Y
    Y = bi_cell.output(H)

    return H, Y
