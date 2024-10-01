#!/usr/bin/env python3
""" This module defines the inception_network function. """
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    This function builds an inception network as described in Going Deeper
    with Convolutions (2014).
    We assume the input data will have shape (224, 224, 3) and use a 'same'
    padding type.
    All convolutions inside the inception block use a rectified linear
    activation (ReLU).
    Returns: the Keras model.
    """
    input_tensor = K.Input(shape=(224, 224, 3))

    def conv_block(input_tensor, filters, kernel_size, strides):
        return K.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                               padding='same', activation='relu',
                               strides=strides)(input_tensor)

    # Convolutional layer
    conv1 = conv_block(input_tensor, 64, 7, 2)
    # Max pooling layer
    max_pool1 = K.layers.MaxPooling2D(pool_size=3,
                                      strides=2,
                                      padding='same')(conv1)

    # Convolutional layers
    conv2 = conv_block(max_pool1, 64, 1, 1)
    conv3 = conv_block(conv2, 192, 3, 1)

    # Max pooling layer
    max_pool2 = K.layers.MaxPooling2D(pool_size=3,
                                      strides=2,
                                      padding='same')(conv3)

    # Inception blocks
    inception3a = inception_block(max_pool2, [64, 96, 128, 16, 32, 32])
    inception3b = inception_block(inception3a, [128, 128, 192, 32, 96, 64])

    # Max pooling layer
    max_pool3 = K.layers.MaxPooling2D(pool_size=3,
                                      strides=2,
                                      padding='same')(inception3b)

    # Inception blocks
    inception4a = inception_block(max_pool3, [192, 96, 208, 16, 48, 64])
    inception4b = inception_block(inception4a, [160, 112, 224, 24, 64, 64])
    inception4c = inception_block(inception4b, [128, 128, 256, 24, 64, 64])
    inception4d = inception_block(inception4c, [112, 144, 288, 32, 64, 64])
    inception4e = inception_block(inception4d, [256, 160, 320, 32, 128, 128])

    # Max pooling layer
    max_pool4 = K.layers.MaxPooling2D(pool_size=3,
                                      strides=2,
                                      padding='same')(inception4e)

    # Inception blocks
    inception5a = inception_block(max_pool4, [256, 160, 320, 32, 128, 128])
    inception5b = inception_block(inception5a, [384, 192, 384, 48, 128, 128])

    # Average pooling layer
    avg_pool = K.layers.AveragePooling2D(pool_size=7, strides=1)(inception5b)

    # Dropout layer
    dropout = K.layers.Dropout(0.4)(avg_pool)

    # Dense layer
    output = K.layers.Dense(1000, activation='softmax')(dropout)

    return K.Model(inputs=input_tensor, outputs=output)
