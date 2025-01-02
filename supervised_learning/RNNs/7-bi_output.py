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

    def backward(self, h_next, x_t):
        """
        This method performs backward propagation for one time step
        Arguments:
            - h_next is a numpy.ndarray of shape (m, h) containing the next
            hidden state. m is the batch size for the data. h is the dimension
            of the hidden state.
            - x_t is a numpy.ndarray of shape (m, i) that contains the data
            input for the cell. m is the batch size for the data and i is the
            dimensionality of the data.
        Returns: h_pev, y
            - h_pev is the previous hidden state
        """
        # Concatenate next hidden state and current input
        concat_input = np.concatenate((h_next, x_t), axis=1)
        # Compute the previous hidden state using tanh activation
        h_pev = np.tanh(np.dot(concat_input, self.Whb) + self.bhb)
        return h_pev

    def output(self, H):
        """
        This method calculates the outputs of the BidirectionalCell
        Arguments:
            - H is a numpy.ndarray of shape (t, m, 2 * h) containing the
            concatenated hidden states from both directions, excluding the
            initial hidden states
                - t is the maximum number of time steps
                - m is the batch size for the data
                - h is the dimensionality of the hidden state
        Returns: Y
            - Y is a numpy.ndarray of shape (t, m, o) containing the outputs
            of the BidirectionalCell
        """
        # Linear transformation followed by softmax
        Z = np.dot(H, self.Wy) + self.by  # Compute logits
        Y = np.exp(Z) / np.sum(np.exp(Z), axis=-1, keepdims=True)  # Softmax
        return Y
