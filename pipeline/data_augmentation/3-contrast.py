#!/usr/bin/env python3
""" This module defines the change_contrast function. """
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    This function randomly adjusts the contrast of an image.
    Args:
        image (tf.Tensor): A 3D tensor representing the input image.
        lower (float): Lower bound for the random contrast factor.
        upper (float): Upper bound for the random contrast factor.
    Returns:
        tf.Tensor: The contrast-adjusted image.
    """
    # Applying random contrast adjustment within the specified range
    adjusted = tf.image.random_contrast(image, lower, upper)
    return adjusted
