#!/usr/bin/env python3
"""This module defines the function create_placeholders."""
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """
    This function returns two placeholders, x and y, for a neural network.
    nx (int): the number of feature columns in our data.
    classes (int): the number of classes in our classifier.
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return x, y
