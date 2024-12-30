#!/usr/bin/env python3
""" This module defines the RNNCell class """
import numpy as np


class RNNCell:
    """ This class represents a cell of a simple RNN """
    def __init__(self, i, h, o):
        """
        Initializer for the RNNCell class
        Arguments:
          - i is the dimensionality of the data
          - h is the dimensionality of the hidden state
          - o is the dimensionality of the outputs
        """
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        This method performs forward propagation for one time step
        Arguments:
            - h_prev is a numpy.ndarray of shape (m, h) containing the previous
            hidden state. m is the batch size for the data. h is the dimension
            of the hidden state.
            - x_t is a numpy.ndarray of shape (m, i) that contains the data
            input for the cell. m is the batch size for the data and i is the
            dimensionality of the data.
        Returns: h_next, y
            - h_next is the next hidden state
            - y is the output of the cell
        """
        h_next = np.tanh(np.matmul(np.hstack((h_prev, x_t)),
                                   self.Wh) + self.bh)
        y = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
        return h_next, y
