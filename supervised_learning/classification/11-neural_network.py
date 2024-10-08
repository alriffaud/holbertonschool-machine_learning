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
        This method initialize the Neural Network.
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

    @staticmethod
    def sigmoid(x: np.ndarray):
        """
        This function calculates value of the sigmoid activation function for
        a value x.
        """
        return 1 / (1 + np.exp(-x))

    def forward_prop(self, X):
        """
        This method calculates the forward propagation of the neural network.
        X (np.ndarray): is the input data (nx, m).
        """
        self.__A1 = self.sigmoid(np.dot(self.__W1, X) + self.__b1)
        self.__A2 = self.sigmoid(np.dot(self.__W2, self.__A1) + self.__b2)
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        This method calculates the cost of the model using logistic regression.
        Y (np.ndarray): is the correct labels for the input data.
        A (np.ndarray): is the activated output of the neuron for each data
                        point.
        """
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(Y * np.log(A) + (1 - Y) *
                                  np.log(1.0000001 - A))
        return cost
