#!/usr/bin/env python3
""" This module defines the NeuralNetwork class. """
import numpy as np


class NeuralNetwork:
    """
    This class defines a neural network with one hidden layer
    performing binary classification.
    """
    def __init__(self, nx, nodes):
        """
        This is the __init__ method.
        nx (int): is the number of input features to the neuron.
        nodes (int): is the number of nodes found in the hidden layer.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        This method returns the weights vector for the hidden layer.
        """
        return (self.__W1)

    @property
    def b1(self):
        """
        This method returns the bias for the hidden layer.
        """
        return (self.__b1)

    @property
    def A1(self):
        """
        This method returns the activated output for the hidden layer.
        """
        return (self.__A1)

    @property
    def W2(self):
        """
        This method returns the weights vector for the output neuron.
        """
        return (self.__W2)

    @property
    def b2(self):
        """
        This method returns the bias for the output neuron.
        """
        return (self.__b2)

    @property
    def A2(self):
        """
        This method returns the activated output for the output neuron.
        """
        return (self.__A2)
