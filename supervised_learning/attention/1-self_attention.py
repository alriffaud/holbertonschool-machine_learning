#!/usr/bin/env python3
"""
Self-Attention layer for machine translation.
"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    SelfAttention computes the attention scores and context vector
    for machine translation tasks based on Bahdanau et al. (2015).
    """

    def __init__(self, units):
        """
        Initialize the SelfAttention layer.

        Args:
            units (int): Number of hidden units in the alignment model.
        """
        super(SelfAttention, self).__init__()

        # Define dense layers for W, U, and V
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Perform the forward pass to compute the attention context and weights.

        Args:
            s_prev (tf.Tensor): Previous decoder hidden state,
                shape (batch, units).
            hidden_states (tf.Tensor): Encoder hidden states,
                shape (batch, input_seq_len, units).

        Returns:
            context (tf.Tensor): Context vector for the decoder,
                shape (batch, units).
            weights (tf.Tensor): Attention weights,
                shape (batch, input_seq_len, 1).
        """
        # Expand s_prev to match the dimensions of hidden_states
        s_prev_exp = tf.expand_dims(s_prev, axis=1)  # Shape: (batch, 1, units)

        # Compute alignment scores using W, U, and V
        score = self.V(tf.nn.tanh(self.W(s_prev_exp) + self.U(hidden_states)))
        # Shape of score: (batch, input_seq_len, 1)

        # Compute attention weights using softmax
        # Shape: (batch, input_seq_len, 1)
        weights = tf.nn.softmax(score, axis=1)

        # Compute the context vector as a weighted sum of hidden states
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        # Shape of context: (batch, units)

        return context, weights
