#!/usr/bin/env python3
""" This module defines the DeepNeuralNetwork class. """
import numpy as np
import pickle


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

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        This method calculates one pass of gradient
        descent on the neural network.
        """
        m = Y.shape[1]
        dZ = cache['A' + str(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            A = cache['A' + str(i - 1)]
            dW = np.matmul(dZ, A.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            W = self.__weights['W' + str(i)]
            dZ = np.matmul(W.T, dZ) * A * (1 - A)
            self.__weights['W' + str(i)] = self.__weights['W' + str(i)] - (
                alpha * dW)
            self.__weights['b' + str(i)] = self.__weights['b' + str(i)] - (
                alpha * db)
        return self.__weights

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        This method trains the deep neural network.
        X (np.ndarray): is the input data.
        Y (np.ndarray): is the correct labels for the input data.
        iterations (int): is the number of iterations to train over.
        alpha (float): is the learning rate.
        verbose (bool): defines whether or not to print information about the
        training.
        graph (bool): defines whether or not to graph information about the
        training once the training has completed.
        step (int): defines the number of iterations between printing
        information.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        This method saves the instance object to a file in pickle format.
        filename (str): is the file to which the object should be saved.
        """
        if not filename.endswith(".pkl"):
            filename = filename + ".pkl"

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load(filename):
        """
        This method loads a pickled DeepNeuralNetwork object.
        filename (str): is the file from which the object should be loaded.
        """
        try:
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None
