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

    @staticmethod
    def sigmoid(x: np.ndarray):
        """
        This function calculates value of the sigmoid activation function for
        a value x.
        """
        return 1 / (1 + np.exp(-x))

    def forward_prop(self, X):
        """
        This method calculates the forward propagation of the neuron.
        """
        self.__A = self.sigmoid(np.dot(self.__W, X) + self.__b)
        return self.__A

    def cost(self, Y, A):
        """
        This method calculates the cost of the model using logistic regression.
        """
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(Y * np.log(A) + (1 - Y) *
                                  np.log(1.0000001 - A))
        return cost