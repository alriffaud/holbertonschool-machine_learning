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
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy
