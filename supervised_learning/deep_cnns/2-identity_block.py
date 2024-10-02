#!/usr/bin/env python3
""" This module defines the identity_block function. """
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    This function builds an identity block as described in Deep Residual
    Learning for Image Recognition (2015).
    All convolutions inside the block are followed by batch normalization
    along the channels axis and a rectified linear activation (ReLU),
    respectively.
    All weights use he normal initialization.
    The seed for the random initialization is set to 0.
    Args:
        A_prev (Keras input): is the output of the previous layer.
        filters (list/tuple): is a tuple or list containing F11, F3, and F12,
            where F11 is the number of filters in the first 1x1 convolution,
            F3 is the number of filters in the 3x3 convolution, and F12 is the
            number of filters in the second 1x1 convolution.
    Returns: the activated output of the identity block.
    """
    F11, F3, F12 = filters

    # Save the input value for the shortcut
    shortcut = A_prev

    # 1x1 convolution
    conv1x1 = K.layers.Conv2D(filters=F11, kernel_size=1, padding='same',
                              kernel_initializer=K.initializers.he_normal(
                                  seed=0))(A_prev)
    # Batch normalization
    batch_norm1 = K.layers.BatchNormalization(axis=3)(conv1x1)
    # Activation
    activation1 = K.layers.Activation('relu')(batch_norm1)

    # 3x3 convolution
    conv3x3 = K.layers.Conv2D(filters=F3, kernel_size=3, padding='same',
                              kernel_initializer=K.initializers.he_normal(
                                  seed=0))(activation1)
    # Batch normalization
    batch_norm2 = K.layers.BatchNormalization(axis=3)(conv3x3)
    # Activation
    activation2 = K.layers.Activation('relu')(batch_norm2)

    # 1x1 convolution
    conv1x1_2 = K.layers.Conv2D(filters=F12, kernel_size=1, padding='same',
                                kernel_initializer=K.initializers.he_normal(
                                    seed=0))(activation2)
    # Batch normalization
    batch_norm3 = K.layers.BatchNormalization(axis=3)(conv1x1_2)

    # Add the input to the output of the block
    add = K.layers.Add()([batch_norm3, shortcut])
    # Activation
    return K.layers.Activation('relu')(add)
