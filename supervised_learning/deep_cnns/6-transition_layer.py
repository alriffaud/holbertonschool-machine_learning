#!/usr/bin/env python3
""" This module defines the transition_layer function. """
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    This function builds a transition layer as described in Densely Connected
    Convolutional Networks (2017).
    Args:
        X (Keras input): the output of the previous layer.
        nb_filters (int): the number of filters in X.
        compression (float): the compression factor for the transition layer.
    Returns: the output of the transition layer and the number of filters
        within the output, respectively.
    """
    he_normal = K.initializers.he_normal(seed=0)
    nb_filters = int(nb_filters * compression)
    # Batch normalization
    batch_norm = K.layers.BatchNormalization(axis=3)(X)
    # Activation
    activation = K.layers.Activation('relu')(batch_norm)
    # Convolution
    conv = K.layers.Conv2D(filters=nb_filters, kernel_size=1, padding='same',
                           kernel_initializer=he_normal)(activation)
    # Average pooling
    avg_pool = K.layers.AveragePooling2D(pool_size=2, strides=2,
                                         padding='same')(conv)
    return avg_pool, nb_filters
