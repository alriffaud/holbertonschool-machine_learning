#!/usr/bin/env python3
"""This module defines the function calculate_accuracy."""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
    This function calculates the accuracy of a prediction.
    Args:
        y (tensor): is a placeholder for the labels of the input data.
        y_pred (tensor): is a tensor containing the network’s predictions.
    """
    if tf.size(y) == 0:
        return 0.0
    correct_predictions = tf.equal(y, y_pred)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy
