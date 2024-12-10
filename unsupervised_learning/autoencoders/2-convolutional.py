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
    for f in reversed(filters[:-1]):
        # Add a Conv2D layer with ReLU activation and 'same' padding
        x = keras.layers.Conv2D(f, (3, 3),
                                activation='relu', padding='same')(x)
        # Add an UpSampling2D layer to upsample the image
        x = keras.layers.UpSampling2D((2, 2))(x)

    # Add the second-to-last convolutional layer with 'valid' padding
    x = keras.layers.Conv2D(filters[-1], (3, 3),
                            activation='relu', padding='valid')(x)

    # Add the final convolutional layer with sigmoid activation
    # (output same as input channels)
    output_img = keras.layers.Conv2D(input_dims[2], (3, 3),
                                     activation='sigmoid', padding='same')(x)

    # ---------------- Models ---------------- #
    # Encoder model: from input image to latent space
    encoder = keras.models.Model(inputs=input_img, outputs=latent_space)

    # Decoder model: from latent space to reconstructed image
    latent_input = keras.layers.Input(shape=latent_dims)
    x = latent_input
    for f in reversed(filters[:-1]):
        x = keras.layers.Conv2D(f, (3, 3), activation='relu',
                                padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(filters[-1], (3, 3),
                            activation='relu', padding='valid')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)  # Fix for shape mismatch
    decoded_output = keras.layers.Conv2D(input_dims[2], (3, 3),
                                         activation='sigmoid',
                                         padding='same')(x)
    decoder = keras.models.Model(inputs=latent_input, outputs=decoded_output)

    # Autoencoder model: combines encoder and decoder
    auto = keras.models.Model(inputs=input_img,
                              outputs=decoder(encoder(input_img)))

    # Compile the autoencoder with Adam optimizer and binary cross-entropy loss
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
