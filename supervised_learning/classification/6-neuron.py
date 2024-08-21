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

    def evaluate(self, X, Y):
        """
        This method evaluates the neuron's predictions.
        """
        A = self.forward_prop(X)
        predictions = np.where(A >= 0.5, 1, 0)
        costs = self.cost(Y, A)
        return predictions, costs

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        This method calculates one pass of gradient descent on the neuron.
        """
        m = X.shape[1]  # Number of examples

        # Calculate the gradient of the weights and bias
        dW = np.dot(A - Y, X.T) / m
        db = np.sum(A - Y) / m

        # Update weights and bias
        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        This method trains the neuron.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)
