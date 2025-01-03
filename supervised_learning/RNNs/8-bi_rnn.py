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
    t, m, _ = X.shape
    h = h_0.shape[1]

    # Initialize hidden states for forward and backward directions
    H_forward = np.zeros((t, m, h))
    H_backward = np.zeros((t, m, h))

    # Forward direction
    h_prev = h_0
    for step in range(t):
        h_prev = bi_cell.forward(h_prev, X[step])
        H_forward[step] = h_prev

    # Backward direction
    h_next = h_t
    for step in reversed(range(t)):
        h_next = bi_cell.backward(h_next, X[step])
        H_backward[step] = h_next

    # Concatenate forward and backward hidden states
    H = np.concatenate((H_forward, H_backward), axis=-1)

    # Compute outputs
    Y = np.array([bi_cell.output(h) for h in H])

    return H, Y
