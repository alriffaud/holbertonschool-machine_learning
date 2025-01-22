#!/usr/bin/env python3
""" This module defines the class RNN Encoder for Machine Translation. """
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    This class represents an RNN Encoder for Machine Translation.
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        This method initialize the RNNEncoder.
        Args:
            vocab (int): Size of the input vocabulary.
            embedding (int): Dimensionality of the embedding vector.
            units (int): Number of hidden units in the RNN cell.
            batch (int): Batch size.
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch  # Store batch size
        self.units = units  # Store number of hidden units

        # Embedding layer to convert word indices to dense vectors
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)

        # GRU layer with Glorot uniform initializer
        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform"
        )

    def initialize_hidden_state(self):
        """
        This method initialize the hidden states for the RNN cell to a tensor
        of zeros.
        Returns:
            Tensor of shape (batch, units): Initialized hidden states.
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        This method forward pass through the encoder.
        Args:
            x (Tensor): Input tensor of shape (batch, input_seq_len),
                containing word indices.
            initial (Tensor): Initial hidden state of shape (batch, units).
        Returns:
            outputs (Tensor): Output of the encoder of shape (batch,
                input_seq_len, units).
            hidden (Tensor): Last hidden state of shape (batch, units).
        """
        # Pass input through the embedding layer
        x = self.embedding(x)

        # Pass embeddings through the GRU layer
        outputs, hidden = self.gru(x, initial_state=initial)

        return outputs, hidden
