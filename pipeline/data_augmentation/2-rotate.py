#!/usr/bin/env python3
""" This module defines the rotate_image function. """
import tensorflow as tf


def rotate_image(image):
    """
    This function rotates an image 90 degrees counter-clockwise.
    Args:
    image (tf.Tensor): A 3D tensor containing the image to rotate.
    Returns:
    tf.Tensor: The rotated image.
    """
    # TensorFlow's rot90 function to rotate the image 90 degrees
    # counter-clockwise
    rotated = tf.image.rot90(image, k=1)
    return rotated
