#!/usr/bin/env python3
"""This module defines the function calculate_loss."""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """
    This function calculates the softmax cross-entropy loss of a prediction.
    Args:
        y (tensor): is a placeholder for the labels of the input data.
        y_pred (tensor): is a tensor containing the network's predictions.
    """
    return tf.losses.softmax_cross_entropy(logits=y_pred, onehot_labels=y)
