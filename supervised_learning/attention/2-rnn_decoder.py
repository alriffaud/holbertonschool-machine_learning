#!/usr/bin/env python3
""" This module defines RNN Decoder for machine translation. """
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    This class decodes input sequences for machine translation using
    a GRU-based RNN and an attention mechanism.
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        This method initialize the RNNDecoder.
        Args:
            vocab (int): Size of the output vocabulary.
            embedding (int): Dimensionality of the embedding vector.
            units (int): Number of hidden units in the RNN cell.
            batch (int): Batch size.
        """
        super(RNNDecoder, self).__init__()

        # Embedding layer to convert word indices to dense vectors
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)

        # GRU layer for processing sequences
        self.gru = tf.keras.layers.GRU(units=units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer="glorot_uniform")

        # Dense layer to map GRU outputs to vocabulary probabilities
        self.F = tf.keras.layers.Dense(units=vocab)

        # SelfAttention layer to compute attention context
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        This method performs the forward pass to decode a single word.
        Args:
            x (tf.Tensor): Previous word in the target sequence as indices
                of shape (batch, 1).
            s_prev (tf.Tensor): Previous decoder hidden state
                of shape (batch, units).
            hidden_states (tf.Tensor): Encoder hidden states
                of shape (batch, input_seq_len, units).
        Returns:
            y (tf.Tensor): Output word as one-hot probabilities
                of shape (batch, vocab).
            s (tf.Tensor): New decoder hidden state
                of shape (batch, units).
        """
        # Compute the attention context and weights
        context, _ = self.attention(s_prev, hidden_states)

        # Expand context to match the dimensions of x for concatenation
        # Shape: (batch, 1, units)
        context_exp = tf.expand_dims(context, axis=1)

        # Pass the input x through the embedding layer
        x_emb = self.embedding(x)  # Shape: (batch, 1, embedding)

        # Concatenate context vector and embedded input along the last axis
        concat_input = tf.concat([context_exp, x_emb], axis=-1)
        # Shape: (batch, 1, units + embedding)

        # Pass the concatenated input through the GRU
        output, s = self.gru(concat_input, initial_state=s_prev)
        # output shape: (batch, 1, units), s shape: (batch, units)

        # Pass the output through the dense layer to compute probabilities
        y = self.F(output)
        # y shape: (batch, 1, vocab)

        # Remove the time dimension from y for the final output
        y = tf.squeeze(y, axis=1)  # Shape: (batch, vocab)

        return y, s
