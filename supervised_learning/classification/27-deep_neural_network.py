#!/usr/bin/env python3
"""
This module contains the class DeepNeuralNetwork
which performs binary classification
"""

import numpy as np
import pickle


class DeepNeuralNetwork:
    """ Defines a Deep Neural Network """

    def __init__(self, nx, layers):
        """ Initializes a Deep Neural Nerwork
                - nx: number of input features.
                - layers: number of nodes in each layer of the network.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) is not list or not layers:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for la in range(self.L):

            if type(layers[la]) is not int or layers[la] < 1:
                raise ValueError("layers must be a list of positive integers")

            inodes = nx if la == 0 else layers[la - 1]

            self.__weights["W" + str(la + 1)] = np.random.randn(
                    layers[la], inodes) * np.sqrt(2 / inodes)

            self.__weights["b" + str(la + 1)] = np.zeros((layers[la], 1))

    @property
    def L(self):
        """ Returns L (array of number of nodes per layer) """
        return self.__L

    @property
    def cache(self):
        """ Returns cache """
        return self.__cache

    @property
    def weights(self):
        """ Returns weights """
        return self.__weights

    @staticmethod
    def __smax(z):
        """ Performs the softmax calculation
            - z: numpy.ndarray with shape (nx, m) that contains the input data
        """
        a = np.exp(z - np.max(z))
        return a / a.sum(axis=0)

    def forward_prop(self, X):
        """ Calculates the forward propagation of the Neural Network
            - X: numpy.ndarray with shape (nx, m) containing the input data.
        """
        self.__cache["A0"] = X

        for i in range(1, self.__L + 1):
            ws = self.__weights["W" + str(i)]
            bs = self.__weights["b" + str(i)]
            A = self.__cache["A" + str(i - 1)]
            Z = ws @ A + bs

            if i < self.__L:
                A = 1 / (1 + np.exp(-Z))
            else:
                A = self.__smax(Z)

            self.__cache["A" + str(i)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using binary cross-entropy.
            - Y: numpy.ndarray with shape (1, m) containing correct labels.
            - A: numpy.ndarray with shape (1, m) containing activated output.
                - m: number of examples.
        """

        return -np.sum(Y * np.log(A)) / Y.shape[1]

    def evaluate(self, X, Y):
        """ Evaluates the neuron prediction and loss
            - X: numpy.ndarray with shape (nx, m) containing the inputs.
            - Y: numpy.ndarray with shape (1, m) containing true labels.
                - m: number of examples.
        """
        m = X.shape[1]
        A = self.forward_prop(X)[0]

        return np.where(A == np.max(A, axis=0), 1, 0), self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Performs the gradient descent calculation
            - Y: numpy.ndarray with shape (1, m) containing true labels.
                - m: number of examples.
            - cache: dictionary containing all the dnn intermediary values.
            - alpha: the learning rate.
        """
        m = Y.shape[1]
        dz = self.__cache["A" + str(self.__L)] - Y

        for i in range(self.__L, 0, -1):
            dw = (1/m) * (dz @ self.__cache["A" + str(i - 1)].T)
            db = (1/m) * np.sum(dz, axis=1, keepdims=True)
            dz = (self.__weights["W" + str(i)].T @ dz) * (
                    self.__cache["A" + str(i - 1)] * (
                        1 - self.__cache["A" + str(i - 1)]))

            self.__weights["W" + str(i)] -= alpha * dw
            self.__weights["b" + str(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """ Trains the dnn and returns the evaluation
                after iterations iterations.
            - X: numpy.ndarray with shape (nx, m) containing the inputs.
            - Y: numpy.ndarray with shape (1, m) containing true labels.
            - iterations: number of iterations to train over.
            - alpha: learning rate.
        """

        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)

        return self.evaluate(X, Y)

    def save(self, filename):
        """ Saves the instance object to a file in pickle format.
                - filename is the file to which the object should be saved.
        """
        if not filename.endswith(".pkl"):
            filename = filename + ".pkl"

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """ Loads a pickled DeepNeuralNetwork object.
            - filename is the file from which the object should be loaded.
        """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
