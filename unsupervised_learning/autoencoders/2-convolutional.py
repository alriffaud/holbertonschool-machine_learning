#!/usr/bin/env python3
""" This module defines the autoencoder function. """
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    This function creates a convolutional autoencoder.
    Args:
        input_dims (tuple): Dimensions of the input data (height, width,
            channels).
        filters (list): List of integers specifying the number of filters in
            each encoder convolutional layer.
        latent_dims (tuple): Dimensions of the latent space representation
            (height, width, channels).
    Returns:
        encoder (Model): The encoder model.
        decoder (Model): The decoder model.
        auto (Model): The full autoencoder model.
    """
    # Input layer for the autoencoder
    input_img = keras.layers.Input(shape=input_dims)

    # ---------------- Encoder ---------------- #
    x = input_img  # Start with the input image
    for f in filters:
        # Add a Conv2D layer with ReLU activation and 'same' padding
        x = keras.layers.Conv2D(f, (3, 3),
                                activation='relu', padding='same')(x)
        # Add a MaxPooling2D layer to downsample the image
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    # At this point, 'x' represents the latent space
    latent_space = x

    # ---------------- Decoder ---------------- #
    latent_input = keras.layers.Input(shape=latent_dims)

    for idx, units in enumerate(reversed(filters)):
        if idx != len(filters) - 1:
            layer = keras.layers.Conv2D(
                filters=units,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                activation="relu",
            )
            if idx == 0:
                outputs = layer(latent_input)
            else:
                outputs = layer(outputs)
        else:
            layer = keras.layers.Conv2D(
                filters=units,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="valid",
                activation="relu",
            )
            outputs = layer(outputs)

        layer = keras.layers.UpSampling2D(size=(2, 2))

        outputs = layer(outputs)

    layer = keras.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="sigmoid",
    )

    outputs = layer(outputs)

    # ---------------- Models ---------------- #
    # Encoder model: from input image to latent space
    encoder = keras.models.Model(inputs=input_img, outputs=latent_space)

    # Decoder model: from latent space to reconstructed image
    decoder = keras.models.Model(inputs=latent_input, outputs=outputs)

    # Autoencoder model: combines encoder and decoder
    auto = keras.models.Model(inputs=input_img,
                              outputs=decoder(encoder(input_img)))

    # Compile the autoencoder
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
