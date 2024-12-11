#!/usr/bin/env python3
""" This module defines the autoencoder function. """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    This function creates a variational autoencoder.
    Args:
        input_dims (int): is an integer containing the dimensions of the model
            input.
        hidden_layers (list): is a list containing the number of nodes for each
            hidden layer in the encoder, respectively. The hidden layers should
            be reversed for the decoder.
        latent_dims (int): is an integer containing the dimensions of the
            latent space representation.
    Returns:
        encoder (Model): The encoder model, which outputs the latent
            representation, mean, and log variance.
        decoder (Model): The decoder model.
        auto (Model): The full autoencoder model.
    """
    # --- ENCODER ---
    # Input layer for the encoder
    inputs = keras.Input(shape=(input_dims,))

    # Building the encoder hidden layers
    x = inputs
    for nodes in hidden_layers:
        # Fully connected layer with ReLU activation
        x = keras.layers.Dense(nodes, activation='relu')(x)

    # Latent space: output mean and log variance for reparameterization trick
    # Mean layer with linear activation
    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    # Log variance layer with linear activation
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    # Reparameterization trick to sample from the latent space
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = keras.backend.random_normal(
            shape=(keras.backend.shape(z_mean)[0], latent_dims))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    # Apply the sampling function
    z = keras.layers.Lambda(
        sampling, output_shape=(latent_dims,))([z_mean, z_log_var])

    # Define the encoder model
    encoder = keras.Model(inputs, [z, z_mean, z_log_var], name="encoder")

    # --- DECODER ---
    # Input layer for the decoder (latent space representation)
    latent_inputs = keras.Input(shape=(latent_dims,))

    # Building the decoder hidden layers (reverse of encoder)
    x = latent_inputs
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    # Output layer to reconstruct the original input
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    # Define the decoder model
    decoder = keras.Model(latent_inputs, outputs, name="decoder")

    # --- FULL AUTOENCODER ---
    # Connect encoder and decoder
    outputs = decoder(encoder(inputs)[0])

    # Define the full autoencoder model
    auto = keras.Model(inputs, outputs, name="autoencoder")

    # Custom loss: binary cross-entropy + KL divergence
    reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss *= input_dims

    kl_loss = -0.5 * keras.backend.sum(
        1 + z_log_var - keras.backend.square(
            z_mean) - keras.backend.exp(z_log_var), axis=-1)
    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)

    # Add the custom loss to the model
    auto.add_loss(vae_loss)

    # Compile the model with Adam optimizer
    auto.compile(optimizer=keras.optimizers.Adam())

    return encoder, decoder, auto
