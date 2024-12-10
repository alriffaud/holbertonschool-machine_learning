#!/usr/bin/env python3
""" This module defines the autoencoder function. """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    This function creates a vanilla autoencoder model.
    Args:
        input_dims (int): is an integer containing the dimensions of the model
            input.
        hidden_layers (list): is a list containing the number of nodes for each
            hidden layer in the encoder, respectively. The hidden layers should
            be reversed for the decoder.
        latent_dims (int): is an integer containing the dimensions of the
            latent space representation.
    Returns:
        tuple: encoder, decoder, autoencoder models.
            encoder is the encoder model.
            decoder is the decoder model.
            auto is the full autoencoder model.
    """
    # Input layer for the encoder
    input_layer = keras.Input(shape=(input_dims,))

    # Building the encoder
    encoded = input_layer
    # Add hidden layers as specified in hidden_layers
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)

    # Adding the latent space layer
    latent_layer = keras.layers.Dense(latent_dims, activation='relu')(encoded)

    # Building the decoder
    decoded = latent_layer
    # Reverse the hidden layers for the decoder
    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)

    # Output layer for the decoder
    output_layer = keras.layers.Dense(input_dims,
                                      activation='sigmoid')(decoded)

    # Creating the encoder model
    encoder = keras.Model(inputs=input_layer, outputs=latent_layer)

    # Input layer for the decoder
    latent_input = keras.Input(shape=(latent_dims,))
    reconstructed = latent_input
    for nodes in reversed(hidden_layers):
        reconstructed = keras.layers.Dense(nodes,
                                           activation='relu')(reconstructed)
    reconstructed = keras.layers.Dense(input_dims,
                                       activation='sigmoid')(reconstructed)

    # Creating the decoder model
    decoder = keras.Model(inputs=latent_input, outputs=reconstructed)

    # Connecting the encoder and decoder to form the autoencoder
    auto_input = keras.Input(shape=(input_dims,))
    encoded_output = encoder(auto_input)
    decoded_output = decoder(encoded_output)
    auto = keras.Model(inputs=auto_input, outputs=decoded_output)

    # Compiling the autoencoder model
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
