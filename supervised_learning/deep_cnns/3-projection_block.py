#!/usr/bin/env python3
""" This module defines the projection_block function. """
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    This function builds a projection block as described in Deep Residual
    Learning for Image Recognition (2015).
    All convolutions inside the block are followed by batch normalization
    along the channels axis and a rectified linear activation (ReLU),
    respectively.
    All weights use he normal initialization.
    Args:
        A_prev (Keras input): the output of the previous layer.
        filters (tuple/list): contains the number of filters in each
        convolution of the block.
            - F11: number of filters in the first 1x1 convolution.
            - F3: number of filters in the 3x3 convolution.
            - F12: number of filters in the second 1x1 convolution.
        s (int): stride of the first convolution  in both the main path and the
        shortcut connection.
    Returns: the activated output of the projection block.
    """
    F11, F3, F12 = filters
    he_normal = K.initializers.he_normal(seed=0)

    # Save the input value for the shortcut
    shortcut = A_prev

    # 1x1 convolution
    conv1x1 = K.layers.Conv2D(filters=F11, kernel_size=1, padding='same',
                              strides=s,
                              kernel_initializer=he_normal)(A_prev)
    # Batch normalization
    batch_norm1 = K.layers.BatchNormalization(axis=3)(conv1x1)
    # Activation
    activation1 = K.layers.Activation('relu')(batch_norm1)

    # 3x3 convolution
    conv3x3 = K.layers.Conv2D(filters=F3, kernel_size=3, padding='same',
                              kernel_initializer=he_normal)(activation1)
    # Batch normalization
    batch_norm2 = K.layers.BatchNormalization(axis=3)(conv3x3)
    # Activation
    activation2 = K.layers.Activation('relu')(batch_norm2)

    # 1x1 convolution
    conv1x1_2 = K.layers.Conv2D(filters=F12, kernel_size=1, padding='same',
                                kernel_initializer=he_normal)(activation2)
    # Batch normalization
    batch_norm3 = K.layers.BatchNormalization(axis=3)(conv1x1_2)

    # Shortcut path
    shortcut = K.layers.Conv2D(filters=F12, kernel_size=1, padding='same',
                               strides=s,
                               kernel_initializer=he_normal)(shortcut)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    # Add the input to the output of the block
    add = K.layers.Add()([batch_norm3, shortcut])
    # Activation
    return K.layers.Activation('relu')(add)
