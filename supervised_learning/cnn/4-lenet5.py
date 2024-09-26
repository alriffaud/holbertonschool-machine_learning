#!/usr/bin/env python3
"""This module defines the lenet5 function."""
import tensorflow.compat.v1 as tf  # type: ignore

tf.disable_v2_behavior()


def lenet5(x, y):
    """
    This function builds a modified version of the LeNet-5 architecture
    using tensorflow.
    The model consists of the following layers in order:
    - Convolutional layer with 6 kernels of shape 5x5 with same padding
    - Max pooling layer with kernels of shape 2x2 with 2x2 strides
    - Convolutional layer with 16 kernels of shape 5x5 with valid padding
    - Max pooling layer with kernels of shape 2x2 with 2x2 strides
    - Fully connected layer with 120 nodes
    - Fully connected layer with 84 nodes
    - Fully connected softmax output layer with 10 nodes
    Args:
        x (tf.placeholder): is a placeholder of shape (m, 28, 28, 1) containing
            the input images for the network.
            - m is the number of images.
        y (tf.placeholder): is a placeholder of shape (m, 10) containing the
            one-hot labels for the network.
            - m is the number of images.
    Returns:
        - a tensor for the softmax activated output
        - a training operation that utilizes Adam optimization (with default
            hyperparameters)
        - a tensor for the loss of the netowrk
        - a tensor for the accuracy of the network
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0)

    # Convolutional layer 1
    conv1 = tf.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=init
    )(x)

    # Max pooling layer 1
    pool1 = tf.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv1)

    # Convolutional layer 2
    conv2 = tf.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation=tf.nn.relu,
        kernel_initializer=init
    )(pool1)

    # Max pooling layer 2
    pool2 = tf.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv2)

    # Flatten the data for the fully connected layers
    flat = tf.layers.Flatten()(pool2)

    # Fully connected layer 1
    fc1 = tf.layers.Dense(
        units=120,
        activation=tf.nn.relu,
        kernel_initializer=init
    )(flat)

    # Fully connected layer 2
    fc2 = tf.layers.Dense(
        units=84,
        activation=tf.nn.relu,
        kernel_initializer=init
    )(fc1)

    # Output layer
    output = tf.layers.Dense(
        units=10,
        kernel_initializer=init
    )(fc2)

    # Loss
    loss = tf.losses.softmax_cross_entropy(y, output)

    # Accuracy
    y_pred = tf.argmax(output, 1)
    y_true = tf.argmax(y, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))

    # Training operation
    train_op = tf.train.AdamOptimizer().minimize(loss)

    return output, train_op, loss, accuracy
