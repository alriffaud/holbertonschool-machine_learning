#!/usr/bin/env python3
""" This module defines the train_model function. """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
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
        learning_rate_decay (bool): indicates whether learning rate decay
        should be used.
        alpha (float): is the initial learning rate.
        decay_rate (float): is the decay rate.
        verbose (bool): is a boolean that determines if output should be
        printed during training.
        shuffle (bool): is is a boolean that determines whether to shuffle the
        batches every epoch.
    Returns: the History object generated after training the model.
    """
    callbacks = []
    if early_stopping:
        callbacks.append(K.callbacks.EarlyStopping(patience=patience))
    if learning_rate_decay:
        def scheduler(epoch):
            """
            Function that takes an epoch index and returns a new learning rate.
            Args:
                epoch (int): is the current epoch.
            Returns: the new learning rate.
            """
            return alpha / (1 + decay_rate * epoch)
        callbacks.append(K.callbacks.LearningRateScheduler(
            scheduler, verbose=1))
    return network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                       validation_data=validation_data, verbose=verbose,
                       shuffle=shuffle, callbacks=callbacks)
