#!/usr/bin/env python3
""" This module defines the change_hue function ."""
import tensorflow as tf


def change_hue(image, delta):
    """
    This function changes the hue of an image.
    Args:
        image (tf.Tensor): A 3D tensor representing the image to adjust.
        delta (float): The amount to add to the hue channel (in the range
            -1.0 to 1.0).
    Returns:
        tf.Tensor: The hue-adjusted image.
    """
    # TensorFlow's adjust_hue function to modify the hue of the image
    altered = tf.image.adjust_hue(image, delta)
    return altered
