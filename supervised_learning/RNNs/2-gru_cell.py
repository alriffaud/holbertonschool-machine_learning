#!/usr/bin/env python3
""" This module defines the GRUCell class """
import numpy as np


class GRUCell:
    """ This class represents a cell of a GRU """
    def __init__(self, i, h, o):
        """
        Initializer for the GRUCell class
        Arguments:
          - i is the dimensionality of the data
          - h is the dimensionality of the hidden state
          - o is the dimensionality of the outputs
        """
        self.Wz = np.random.randn(h + i, h)
        self.Wr = np.random.randn(h + i, h)
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
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
        h_x = np.hstack((h_prev, x_t))
        z = np.dot(h_x, self.Wz) + self.bz
        z = 1 / (1 + np.exp(-z))
        r = np.dot(h_x, self.Wr) + self.br
        r = 1 / (1 + np.exp(-r))
        h_xr = np.hstack((r * h_prev, x_t))
        h_tilde = np.tanh(np.dot(h_xr, self.Wh) + self.bh)
        h_next = (1 - z) * h_prev + z * h_tilde
        y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
        return h_next, y
