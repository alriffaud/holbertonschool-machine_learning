#!/usr/bin/env python3
""" This module defines the optimize_model function. """
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    This function optimization for a keras model with categorical crossentropy
    loss and accuracy metrics.
    Args:
        network (model): is the model to optimize.
        alpha (float): is the learning rate.
        beta1 (float): is the first Adam optimization parameter.
        beta2 (float): is the second Adam optimization parameter.
    Returns: None
    """
    network.compile(optimizer=K.optimizers.Adam(alpha, beta1, beta2),
                    loss='categorical_crossentropy', metrics=['accuracy'])
    return None
