#!/usr/bin/env python3
""" This module defines the flip_image function """
import tensorflow as tf

def flip_image(image):
    """
    This function flips an image horizontally.
    Args:
    image (tf.Tensor): A 3D tensor representing the image to flip.
    Returns:
    tf.Tensor: The horizontally flipped image.
    """
    # Use TensorFlow's image flipping function to reverse the image along
    # the width axis
    flipped = tf.image.flip_left_right(image)
    return flipped
