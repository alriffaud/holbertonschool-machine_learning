#!/usr/bin/env python3
""" This module defines the inception_block function. """
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    This function builds an inception block as described in Going Deeper with
    Convolutions (2014).
    All convolutions inside the inception block use a rectified linear
    activation (ReLU).
    Args:
        A_prev (Keras input): the output from the previous layer.
        filters (list/tuple): contains the number of filters for each
            convolution in the block. It should be ordered as follows:
            F1: number of filters in the 1x1 convolution.
            F3R: number of filters in the 1x1 convolution before the 3x3
                convolution.
            F3: number of filters in the 3x3 convolution.
            F5R: number of filters in the 1x1 convolution before the 5x5
                convolution.
            F5: number of filters in the 5x5 convolution.
            FPP: number of filters in the 1x1 convolution after the max pooling.
    Returns: the concatenated output of the inception block.
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # 1x1 convolution
    conv1x1 = K.layers.Conv2D(filters=F1, kernel_size=1, padding='same',
                              activation='relu')(A_prev)

    # 1x1 convolution before 3x3 convolution
    conv3x3 = K.layers.Conv2D(filters=F3R, kernel_size=1, padding='same',
                              activation='relu')(A_prev)
    # 3x3 convolution
    conv3x3 = K.layers.Conv2D(filters=F3, kernel_size=3, padding='same',
                              activation='relu')(conv3x3)

    # 1x1 convolution before 5x5 convolution
    conv5x5 = K.layers.Conv2D(filters=F5R, kernel_size=1, padding='same',
                              activation='relu')(A_prev)
    # 5x5 convolution
    conv5x5 = K.layers.Conv2D(filters=F5, kernel_size=5, padding='same',
                              activation='relu')(conv5x5)

    # Max pooling
    max_pool = K.layers.MaxPooling2D(pool_size=3, strides=1, padding='same')(A_prev)
    # 1x1 convolution after max pooling
    max_pool = K.layers.Conv2D(filters=FPP, kernel_size=1, padding='same',
                              activation='relu')(max_pool)

    # Concatenate the outputs
    return K.layers.concatenate([conv1x1, conv3x3, conv5x5, max_pool])
