#!/usr/bin/env python3
""" This module defines the build_model function. """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    This function builds a neural network with the Keras library.
    Args:
        nx (int):  is the number of input features to the network.
        layers (list): contains the number of nodes in each layer of the
        network.
        activations (list): contains the activation functions used for each
        layer of the network.
        lambtha (float): is the L2 regularization parameter.
        keep_prob (float): is the probability that a node will be kept for
        dropout.
    """
    # The Keras sequential model is initialized
    model = K.Sequential()
    # L2 regularization is set up
    l2 = K.regularizers.l2(lambtha)
    # Layers are added to the model
    for i in range(len(layers)):
        # If it is the first layer, the input shape is specified
        if i == 0:
            model.add(K.layers.Dense(layers[i], activation=activations[i],
                                     kernel_regularizer=l2, input_shape=(nx,)))
        else:
            # A dense layer with L2 regularization is added
            model.add(K.layers.Dense(layers[i], activation=activations[i],
                                     kernel_regularizer=l2))
        # A dropout layer is added if it is not the last layer
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
