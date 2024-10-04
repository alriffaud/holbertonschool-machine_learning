#!/usr/bin/env python3
""" This module defines the densenet121 function. """
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    This function builds the DenseNet-121 architecture as described in Densely
    Connected Convolutional Networks (2017).
    All convolutions inside the blocks should be preceded by Batch
    Normalization and a rectified linear activation (ReLU), respectively.
    All weights should use he normal initialization.
    Args:
        growth_rate (int): the growth rate for the dense blocks.
        compression (float): the compression factor for the transition layers.
    Returns: the Keras model.
    """
    he_normal = K.initializers.he_normal(seed=0)
    X = K.Input(shape=(224, 224, 3))
    # Initial convolution
    batch_norm1 = K.layers.BatchNormalization(axis=3)(X)
    activation1 = K.layers.Activation('relu')(batch_norm1)
    conv1 = K.layers.Conv2D(filters=2 * growth_rate, kernel_size=7,
                            padding='same', strides=2,
                            kernel_initializer=he_normal)(activation1)
    max_pool1 = K.layers.MaxPooling2D(pool_size=3, strides=2,
                                      padding='same')(conv1)
    # Dense block 1
    dense1, nb_filters = dense_block(max_pool1, 2 * growth_rate,
                                     growth_rate, 6)
    # Transition layer 1
    trans1, nb_filters = transition_layer(dense1, nb_filters, compression)
    # Dense block 2
    dense2, nb_filters = dense_block(trans1, nb_filters, growth_rate, 12)
    # Transition layer 2
    trans2, nb_filters = transition_layer(dense2, nb_filters, compression)
    # Dense block 3
    dense3, nb_filters = dense_block(trans2, nb_filters, growth_rate, 24)
    # Transition layer 3
    trans3, nb_filters = transition_layer(dense3, nb_filters, compression)
    # Dense block 4
    dense4, nb_filters = dense_block(trans3, nb_filters, growth_rate, 16)
    # Classification layer
    avg_pool = K.layers.AveragePooling2D(pool_size=7, strides=None,
                                         padding='same')(dense4)
    dense = K.layers.Dense(units=1000, activation='softmax',
                           kernel_initializer=he_normal)(avg_pool)
    model = K.models.Model(inputs=X, outputs=dense)
    return model
