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

    def evaluate(self, X, Y):
        """
        This method evaluates the neural network's predictions.
        X (np.ndarray): is the input data (nx, m).
        Y (np.ndarray): is the correct labels for the input data.
        """
        self.forward_prop(X)
        predictions = np.where(self.__A2 >= 0.5, 1, 0)
        costs = self.cost(Y, self.__A2)
        return predictions, costs

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        This method calculates one pass of gradient
        descent on the neural network.
        """
        m = Y.shape[1]  # Number of examples

        # Calculate the gradient of the weights and bias
        dz2 = A2 - Y
        dW2 = (1 / m) * np.dot(dz2, A1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.dot(self.__W2.T, dz2) * A1 * (1 - A1)
        dW1 = (1 / m) * np.dot(dz1, X.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

        # Update weights and bias
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1

        return self.__W1, self.__b1, self.__W2, self.__b2

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        This method trains the neural network.
        X (np.ndarray): is the input data (nx, m).
        Y (np.ndarray): is the correct labels for the input data.
        iterations (int): is the number of iterations to train over.
        alpha (float): is the learning rate.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)

        return self.evaluate(X, Y)
