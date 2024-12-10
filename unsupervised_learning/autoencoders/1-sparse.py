#!/usr/bin/env python3
""" This module defines the autoencoder function. """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    This function creates a sparse autoencoder model.
    Args:
        input_dims (int): is an integer containing the dimensions of the model
            input.
        hidden_layers (list): is a list containing the number of nodes for each
            hidden layer in the encoder, respectively. The hidden layers should
            be reversed for the decoder.
        latent_dims (int): is an integer containing the dimensions of the
            latent space representation.
        lambtha (float): is the regularization parameter used for L1
            regularization.
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
    for nodes in hidden_layers:
        # Add dense layers with ReLU activation for the encoder
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)

    # Add the latent layer with L1 regularization
    latent_layer = keras.layers.Dense(
        latent_dims,
        activation='relu',
        # L1 regularization on latent activations
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(encoded)

    # Building the decoder
    decoded = latent_layer
    for nodes in reversed(hidden_layers):
        # Add dense layers with ReLU activation for the decoder
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)

    # Output layer for the decoder uses sigmoid activation
    output_layer = keras.layers.Dense(input_dims,
                                      activation='sigmoid')(decoded)

    # Create the encoder model
    encoder = keras.Model(inputs=input_layer, outputs=latent_layer)

    # Create the decoder model
    latent_input = keras.Input(shape=(latent_dims,))
    reconstructed = latent_input
    for nodes in reversed(hidden_layers):
        # Add dense layers with ReLU activation for the decoder
        reconstructed = keras.layers.Dense(nodes,
                                           activation='relu')(reconstructed)
    reconstructed = keras.layers.Dense(input_dims,
                                       activation='sigmoid')(reconstructed)
    decoder = keras.Model(inputs=latent_input, outputs=reconstructed)

    # Create the full autoencoder model by connecting encoder and decoder
    auto_input = keras.Input(shape=(input_dims,))
    encoded_output = encoder(auto_input)
    decoded_output = decoder(encoded_output)
    auto = keras.Model(inputs=auto_input, outputs=decoded_output)

    # Compile the autoencoder with Adam optimizer and binary
    # cross-entropy loss
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
