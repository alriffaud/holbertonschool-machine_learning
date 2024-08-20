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
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        This method returns the  weights vector for the neuron.
        """
        return (self.__W)

    @property
    def b(self):
        """
        This method returns the bias for the neuron.
        """
        return (self.__b)

    @property
    def A(self):
        """
        This method returns the activated output of the neuron.
        """
        return (self.__A)
