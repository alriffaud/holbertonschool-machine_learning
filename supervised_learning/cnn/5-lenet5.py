#!/usr/bin/env python3
""" This module defines the lenet5 function. """
from tensorflow import keras as K


def lenet5(X):
    """
    This function builds a modified version of the LeNet-5 architecture
    using keras.
    The model consists of the following layers in order:
    - Convolutional layer with 6 kernels of shape 5x5 with same padding
    - Max pooling layer with kernels of shape 2x2 with 2x2 strides
    - Convolutional layer with 16 kernels of shape 5x5 with valid padding
    - Max pooling layer with kernels of shape 2x2 with 2x2 strides
    - Fully connected layer with 120 nodes
    - Fully connected layer with 84 nodes
    - Fully connected softmax output layer with 10 nodes
    Args:
        X (K.Input): is a tensor of shape (m, 28, 28, 1) containing the input
            images for the network.
            - m is the number of images.
    Returns: a K.Model compiled to use Adam optimization (with default
        hyperparameters) and accuracy metrics.
    """
    init = K.initializers.he_normal(seed=0)

    # Convolutional layer 1
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(X)

    # Max pooling layer 1
    pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv1)

    # Convolutional layer 2
    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=init
    )(pool1)

    # Max pooling layer 2
    pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv2)

    # Flatten the data for the fully connected layers
    flatten = K.layers.Flatten()(pool2)

    # Fully connected layer 1
    FC1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=init
    )(flatten)

    # Fully connected layer 2
    FC2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=init
    )(FC1)

    # Fully connected softmax output layer
    output = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=init
    )(FC2)

    model = K.Model(inputs=X, outputs=output)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
