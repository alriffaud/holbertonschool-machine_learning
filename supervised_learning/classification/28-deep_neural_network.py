#!/usr/bin/env python3
""" This module defines the DeepNeuralNetwork class. """
import numpy as np
import pickle


class DeepNeuralNetwork:
    """
    This class defines a deep neural network performing binary classification.
    """
    def __init__(self, nx, layers, activation='sig'):
        """
        This method initializes the DeepNeuralNetwork class.
        nx (int): is the number of input features.
        layers (list): is a list representing the number of nodes in each
        layer of the network.
        activation (str): represents the type of activation function used in
        the hidden layers.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if activation not in ('sig', 'tanh'):
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation
        for i in range(self.__L):
            if i == 0:
                prev_layer = nx
            else:
                prev_layer = layers[i - 1]
            self.__weights[f"W{i + 1}"] = \
                np.random.randn(layers[i], prev_layer) * \
                np.sqrt(2 / prev_layer)
            self.__weights[f"b{i + 1}"] = np.zeros((layers[i], 1))

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

    @property
    def activation(self):
        """
        This method returns the value stored in the activation function.
        """
        return self.__activation

    @staticmethod
    def __smax(z):
        """
        Performs the softmax calculation
        z: numpy.ndarray with shape (nx, m) that contains the input data
        """
        a = np.exp(z - np.max(z, axis=0, keepdims=True))
        return a / np.sum(a, axis=0, keepdims=True)

    def forward_prop(self, X):
        """
        This method calculates the forward propagation of the neural network.
        X (np.ndarray): is the input data (number X, number examples).
        """
        self.__cache["A0"] = X
        for i in range(1, self.__L + 1):
            ws = self.__weights["W" + str(i)]
            bs = self.__weights["b" + str(i)]
            A = self.__cache["A" + str(i - 1)]
            Z = ws @ A + bs
            if i < self.__L:
                if self.__activation == 'sig':
                    A = 1 / (1 + np.exp(-Z))
                else:
                    A = np.tanh(Z)
            else:
                A = self.__smax(Z)
            self.__cache["A" + str(i)] = A
        return A, self.__cache

    def cost(self, Y, A):
        """
        This method calculates the cost of the model using logistic regression.
        Y (np.ndarray): is the correct labels for the input data.
        A (np.ndarray): is the predicted labels.
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A))
        return cost / m

    def evaluate(self, X, Y):
        """
        This method evaluates the neural network's predictions.
        X (np.ndarray): is the input data (number X, number examples).
        Y (np.ndarray): is the correct labels for the input data.
        """
        m = X.shape[1]
        A = self.forward_prop(X)[0]
        return np.where(A == np.max(A, axis=0), 1, 0), self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        This method calculates one pass of gradient
        descent on the neural network.
        """
        m = Y.shape[1]
        dZ = cache['A' + str(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            prev_A = cache["A" + str(i - 1)]
            dW = (1 / m) * np.matmul(dZ, prev_A.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            if self.__activation == 'sig':
                dZ = np.matmul(self.__weights[f"W{i}"].T, dZ) * prev_A * \
                    (1 - prev_A)
            else:
                dZ = np.matmul(self.__weights[f"W{i}"].T, dZ) * \
                    (1 - (prev_A ** 2))
            self.__weights['W' + str(i)] -= alpha * dW
            self.__weights['b' + str(i)] -= alpha * db
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
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        costs = []
        count = []
        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            if i != iterations:
                self.gradient_descent(Y, self.cache, alpha)
            cost = self.cost(Y, A)
            costs.append(cost)
            count.append(i)
            if verbose and (i % step == 0 or i == 0 or i == iterations):
                print("Cost after {} iterations: {}".format(i, cost))
        if graph:
            import matplotlib.pyplot as plt
            plt.plot(count, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
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
