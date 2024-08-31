#!/usr/bin/env python3
"""This module defines the train function."""
import tensorflow.compat.v1 as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"):
    """
    This function trains, and saves a neural network classifier.
    Args:
        X_train (numpy.ndarray): contains the training input data.
        Y_train (numpy.ndarray): contains the training labels.
        X_valid (numpy.ndarray): contains the validation input data.
        Y_valid (numpy.ndarray): contains the validation labels.
        layer_sizes (list): contains the number of nodes in each layer of the
        network.
        activations (list): contains the activation functions for each layer of
        the network.
        alpha (float): is the learning rate
        iterations (int): is the number of iterations to train over
        save_path (str):designates where to save the model. Defaults to
        "/tmp/model.ckpt".
    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)
    # Add operations and tensors to collections for later access
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('train_op', train_op)
    tf.add_to_collection('accuracy', accuracy)
    # Initialize TensorFlow variables
    init = tf.global_variables_initializer()
    # Create an object to save the model
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)  # Initialize all variables
        for i in range(iterations + 1):
            train_loss, train_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_loss, valid_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
            # Print metrics after every 100 iterations, the 0th & the last
            if i % 100 == 0 or i == iterations:
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {train_loss}")
                print(f"\tTraining Accuracy: {train_accuracy}")
                print(f"\tValidation Cost: {valid_loss}")
                print(f"\tValidation Accuracy: {valid_accuracy}")
            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})
        # Save the trained model
        return saver.save(sess, save_path)
