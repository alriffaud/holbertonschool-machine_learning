#!/usr/bin/env python3
""" This module defines the dense_block function. """
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    This function builds a dense block as described in Densely Connected
    Convolutional Networks (2017).
    We use bottleneck layers used for DenseNet-B.
    All convolutions should be preceded by Batch Normalization and a
    rectified linear activation (ReLU), respectively.
    Args:
        X (Keras input): the output of the previous layer.
        nb_filters (int): the number of filters in X.
        growth_rate (int): the growth rate for the dense block.
        layers (int): the number of layers in the dense block.
    Returns: the concatenated output of each layer within the dense block and
        the number of filters within the concatenated outputs, respectively.
    """
    he_normal = K.initializers.he_normal(seed=0)
    concat = X
    for i in range(layers):
        # 1x1 convolution
        conv1x1 = K.layers.BatchNormalization(axis=3)(concat)
        conv1x1 = K.layers.Activation('relu')(conv1x1)
        conv1x1 = K.layers.Conv2D(filters=4*growth_rate, kernel_size=1,
                                  padding='same',
                                  kernel_initializer=he_normal)(conv1x1)
        # 3x3 convolution
        conv3x3 = K.layers.BatchNormalization(axis=3)(conv1x1)
        conv3x3 = K.layers.Activation('relu')(conv3x3)
        conv3x3 = K.layers.Conv2D(filters=growth_rate, kernel_size=3,
                                  padding='same',
                                  kernel_initializer=he_normal)(conv3x3)
        # Concatenate the output of each layer within the dense block
        concat = K.layers.concatenate([concat, conv3x3])
        nb_filters += growth_rate
    return concat, nb_filters
