#!/usr/bin/env python3
""" This module defines the crop_image function """
import tensorflow as tf


def crop_image(image, size):
    """
    This function performs a random crop of an image.
    Args:
        image (tf.Tensor): A 3D tensor containing the image to crop.
        size (tuple): A tuple containing the desired output size of the
            crop (height, width, channels).
    Returns:
    tf.Tensor: The randomly cropped image with the specified size.
    """
    # TensorFlow's random_crop function to extract a random patch of
    # the image
    cropped = tf.image.random_crop(image, size)
    return cropped
