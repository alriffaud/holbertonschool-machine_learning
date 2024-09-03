#!/usr/bin/env python3
""" This module defines the train_model function. """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, verbose=True,
                shuffle=False):
    """
    This function trains a model using mini-batch gradient descent.
    Args:
        network (model): is the model to optimize.
        data (numpy.ndarray): numpy array of shape (m, nx) containing the input
        data.
        labels (numpy.ndarra): numpy array of shape (m, classes) containing the
        labels of data.
        batch_size (int): is the size of the batch used for mini-batch gradient
        descent.
        epochs (int): is the number of passes through data for mini-batch
        gradient descent.
        verbose (bool): is a boolean that determines if output should be
        printed during training.
        shuffle (bool): is is a boolean that determines whether to shuffle the
        batches every epoch.
    Returns: the History object generated after training the model.
    """
    return network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                       verbose=verbose, shuffle=shuffle)
