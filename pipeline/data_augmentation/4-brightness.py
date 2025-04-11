#!/usr/bin/env python3
""" This module defines the change_brightness function. """
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    This function randomly changes the brightness of an image.
    Args:
        image (tf.Tensor): A 3D tensor representing the image to be adjusted.
        max_delta (float): Maximum delta for brightness adjustment (the image
            will be adjusted by a random value in the range [-max_delta,
            max_delta]).
    Returns:
        tf.Tensor: The brightness-altered image.
    """
    # TensorFlow's random_brightness function to randomly change the
    # brightness of the image.
    altered = tf.image.random_brightness(image, max_delta)
    return altered
