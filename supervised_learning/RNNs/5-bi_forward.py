#!/usr/bin/env python3
""" This module defines the BidirectionalCell class """
import numpy as np


class BidirectionalCell:
    """ This class represents a bidirectional cell of an RNN """
    def __init__(self, i, h, o):
        """
        Initializer for the BidirectionalCell class
        Arguments:
          - i is the dimensionality of the data
          - h is the dimensionality of the hidden state
          - o is the dimensionality of the outputs
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.bhf = np.zeros((1, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.bhb = np.zeros((1, h))
        self.Wy = np.random.normal(size=(2 * h, o))
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
        """
        # Concatenate previous hidden state and current input
        concat_input = np.concatenate((h_prev, x_t), axis=1)
        # Compute the next hidden state using tanh activation
        h_next = np.tanh(np.dot(concat_input, self.Whf) + self.bhf)
        return h_next
