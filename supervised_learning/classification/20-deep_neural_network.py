#!/usr/bin/env python3
""" This module defines the DeepNeuralNetwork class. """
import numpy as np


class DeepNeuralNetwork:
    """
    This class defines a deep neural network performing binary classification.
    """
    def __init__(self, nx, layers):
        """
        This method initializes the DeepNeuralNetwork class.
        nx (int): is the number of input features.
        layers (list): is a list representing the number of nodes in each
        layer of the network.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.__L):
            if not (isinstance(layers[i], int) and layers[i] > 0):
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                self.__weights['W1'] = (np.random.randn(layers[i], nx)
                                        * np.sqrt(2. / nx))
            else:
                self.__weights['W' + str(i + 1)] = (np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2. / layers[i - 1]))
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """
        This method returns the number of layers of the deep neural network.
        """
        return self.__L

    @property
    def cache(self):
        """
        This method returns the values stored in cache dictionary.
        """
        return self.__cache

    @property
    def weights(self):
        """
        This method returns the values stored in the weights dictionary.
        """
        return self.__weights

    def forward_prop(self, X):
        """
        This method calculates the forward propagation of the neural network.
        X (np.ndarray): is the input data (number X, number examples).
        """
        self.__cache['A0'] = X
        for i in range(self.__L):
            W = self.__weights['W' + str(i + 1)]
            b = self.__weights['b' + str(i + 1)]
            Z = np.matmul(W, self.__cache['A' + str(i)]) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache['A' + str(i + 1)] = A
        return A, self.__cache

    def cost(self, Y, A):
        """
        This method calculates the cost of the model using logistic regression.
        Y (np.ndarray): is the correct labels for the input data.
        A (np.ndarray): is the predicted labels.
        """
        m = Y.shape[1]
        cost = -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        return cost / m

    def evaluate(self, X, Y):
        """
        This method evaluates the neural network's predictions.
        X (np.ndarray): is the input data (number X, number examples).
        Y (np.ndarray): is the correct labels for the input data.
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        A = np.where(A >= 0.5, 1, 0)
        return A, cost
