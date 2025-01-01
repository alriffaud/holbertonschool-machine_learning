#!/usr/bin/env python3
""" This module defines the LSTMCell class """
import numpy as np


class LSTMCell:
    """ This class represents a cell of an LSTM """
    def __init__(self, i, h, o):
        """
        Initializer for the LSTMCell class
        Arguments:
          - i is the dimensionality of the data
          - h is the dimensionality of the hidden state
          - o is the dimensionality of the outputs
        """
        self.Wf = np.random.randn(h + i, h)
        self.Wu = np.random.randn(h + i, h)
        self.Wc = np.random.randn(h + i, h)
        self.Wo = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        This method performs forward propagation for one time step
        Arguments:
            - h_prev is a numpy.ndarray of shape (m, h) containing the previous
            hidden state. m is the batch size for the data. h is the dimension
            of the hidden state.
            - c_prev is a numpy.ndarray of shape (m, h) containing the previous
            cell state. m is the batch size for the data. h is the dimension
            of the hidden state.
            - x_t is a numpy.ndarray of shape (m, i) that contains the data
            input for the cell. m is the batch size for the data and i is the
            dimensionality of the data.
        Returns: h_next, c_next, y
            - h_next is the next hidden state
            - c_next is the next cell state
            - y is the output of the cell
        """
        h_x = np.hstack((h_prev, x_t))
        f = np.dot(h_x, self.Wf) + self.bf
        f = 1 / (1 + np.exp(-f))
        u = np.dot(h_x, self.Wu) + self.bu
        u = 1 / (1 + np.exp(-u))
        o = np.dot(h_x, self.Wo) + self.bo
        o = 1 / (1 + np.exp(-o))
        c_tilde = np.tanh(np.dot(h_x, self.Wc) + self.bc)
        c_next = f * c_prev + u * c_tilde
        h_next = o * np.tanh(c_next)
        y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
        return h_next, c_next, y
