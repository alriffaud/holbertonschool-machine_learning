#!/usr/bin/env python3
"""This module defines the evaluate function."""
import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """
    This function evaluates the output of a neural network.
    Args:
        X (numpy.ndarray): contains the input data to evaluate.
        Y (numpy.ndarray): contains the one-hot labels for X.
        save_path (_type_): is the location to load the model from.
    Returns: the network's prediction, accuracy, and loss, respectively.
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        feed_dict = {x: X, y: Y}
        prediction = sess.run(y_pred, feed_dict)
        loss_value = sess.run(loss, feed_dict)
        accuracy_value = sess.run(accuracy, feed_dict)
    return prediction, accuracy_value, loss_value
