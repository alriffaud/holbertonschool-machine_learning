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
    # The input layer is defined with the shape of the input data
    inputs = K.layers.Input(shape=(nx,))
    # L2 regularization is set up
    l2 = K.regularizers.l2(lambtha)
    # A dense layer is defined that connects to the input
    outputs = K.layers.Dense(layers[0], activation=activations[0],
                             kernel_regularizer=l2)(inputs)
    # Dropout is added to the first layer
    outputs = K.layers.Dropout(1 - keep_prob)(outputs)
    # Layers are added to the model
    for i in range(1, len(layers)):
        # A dense layer with L2 regularization is added
        outputs = K.layers.Dense(layers[i], activation=activations[i],
                                 kernel_regularizer=l2)(outputs)
        # A dropout layer is added if it is not the last layer
        if i < len(layers) - 1:
            outputs = K.layers.Dropout(1 - keep_prob)(outputs)
    # The model is created by specifying inputs and outputs
    model = K.models.Model(inputs=inputs, outputs=outputs)
    return model
