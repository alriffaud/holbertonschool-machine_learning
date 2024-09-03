#!/usr/bin/env python3
""" This module defines the train_model function. """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """
    This function trains a model using mini-batch gradient descent to also
    analyze validaiton data.
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
        validation_data (tuple): is the data to validate the model with, if not
        None.
        early_stopping (bool): indicates whether early stopping should be used.
        patience (int): is the patience used for early stopping.
        verbose (bool): is a boolean that determines if output should be
        printed during training.
        shuffle (bool): is is a boolean that determines whether to shuffle the
        batches every epoch.
    Returns: the History object generated after training the model.
    """
    if early_stopping:
        callbacks = [K.callbacks.EarlyStopping(patience=patience)]
    else:
        callbacks = None
    return network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                       validation_data=validation_data, verbose=verbose,
                       shuffle=shuffle, callbacks=callbacks)
