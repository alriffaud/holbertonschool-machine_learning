#!/usr/bin/env python3
""" This module defines the rnn function. """
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    This function performs forward propagation for a simple RNN
    Arguments:
      - rnn_cell is an instance of RNNCell that represents the cell where
        the RNN will be run
      - X is a numpy.ndarray of shape (t, m, i) that contains the data to be
        used
        - t is the maximum number of time steps
        - m is the batch size
        - i is the dimensionality of the data
      - h_0 is a numpy.ndarray of shape (m, h) containing the initial hidden
        state
        - h is the dimensionality of the hidden state
    Returns: H, Y
      - H is a numpy.ndarray containing all of the hidden states
      - Y is a numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    h = h_0.shape[1]
    # Initialize H and Y to store hidden states and outputs
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, rnn_cell.Wy.shape[1]))
    H[0] = h_0
    # Perform forward propagation for each time step
    for step in range(t):
        H[step + 1], Y[step] = rnn_cell.forward(H[step], X[step])
    return H, Y
