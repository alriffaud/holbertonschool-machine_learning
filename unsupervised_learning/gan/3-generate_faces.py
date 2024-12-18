#!/usr/bin/env python3
""" This module defines the convolutional_GenDiscr function. """
import tensorflow as tf
from tensorflow.keras import layers, Model


def convolutional_GenDiscr():
    """ Builds a Convolutional GAN """

    def generator():
        """ Builds the generator model """
        input_layer = layers.Input(shape=(16,))

        x = layers.Dense(2048)(input_layer)
        x = layers.Reshape((2, 2, 512))(x)

        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('tanh')(x)

        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(16, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('tanh')(x)

        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(1, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        output_layer = layers.Activation('tanh')(x)

        return Model(inputs=input_layer,
                     outputs=output_layer, name="generator")

    def get_discriminator():
        """ Builds the discriminator model """
        input_layer = layers.Input(shape=(16, 16, 1))

        x = layers.Conv2D(32, (3, 3), padding='same')(input_layer)
        x = layers.MaxPooling2D()(x)
        x = layers.Activation('tanh')(x)

        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Activation('tanh')(x)

        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Activation('tanh')(x)

        x = layers.Conv2D(256, (3, 3), padding='same')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Activation('tanh')(x)

        x = layers.Flatten()(x)
        output_layer = layers.Dense(1)(x)

        return Model(inputs=input_layer,
                     outputs=output_layer, name="discriminator")

    return generator(), get_discriminator()
