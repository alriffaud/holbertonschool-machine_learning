#!/usr/bin/env python3
""" This module defines the Neuron class. """
import numpy as np


class Neuron:
    """ This Neuron defines a single neuron performing binary
    classification. """
    def __init__(self, nx):
        """
        This is the __init__ method.
        nx (int): is the number of input features to the neuron.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
