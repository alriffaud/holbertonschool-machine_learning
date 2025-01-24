#!/usr/bin/env python3
""" This module defines the class MultiHeadAttention for a transformer. """
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    This class represents the Multi-Head Attention mechanism for transformers.
    """
    def __init__(self, dm, h):
        """
        This method initializes the MultiHeadAttention layer.
        Args:
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads. Must divide dm.
        """
        super(MultiHeadAttention, self).__init__()
        if dm % h != 0:
            raise ValueError("dm must be divisible by h")

        self.dm = dm
        self.h = h
        self.depth = dm // h  # Depth of each head

        # Dense layers to generate Q, K, V
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)

        # Dense layer to project the concatenated output
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        This method splits the last dimension into (h, depth) and transposes to
        shape (batch, h, seq_len, depth).
        Args:
            x (tf.Tensor): Tensor to split and reshape.
            batch_size (int): Batch size of the input.
        Returns:
            tf.Tensor: Reshaped tensor with shape (batch, h, seq_len, depth).
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        This method computes the multi-head attention.
        Args:
            Q (tf.Tensor): Query tensor, shape (batch, seq_len_q, dm).
            K (tf.Tensor): Key tensor, shape (batch, seq_len_v, dm).
            V (tf.Tensor): Value tensor, shape (batch, seq_len_v, dm).
            mask (tf.Tensor): Mask tensor, shape (batch, seq_len_q, seq_len_v).
        Returns:
            tf.Tensor: Output tensor of shape (batch, seq_len_q, dm).
            tf.Tensor: Attention weights of shape (batch, h, seq_len_q,
                seq_len_v).
        """
        batch_size = tf.shape(Q)[0]

        # Step 1: Project Q, K, V using the dense layers
        Q = self.Wq(Q)  # Shape: (batch, seq_len_q, dm)
        K = self.Wk(K)  # Shape: (batch, seq_len_v, dm)
        V = self.Wv(V)  # Shape: (batch, seq_len_v, dm)

        # Step 2: Split Q, K, V into multiple heads
        # Shape: (batch, h, seq_len_q, depth)
        Q = self.split_heads(Q, batch_size)
        # Shape: (batch, h, seq_len_v, depth)
        K = self.split_heads(K, batch_size)
        # Shape: (batch, h, seq_len_v, depth)
        V = self.split_heads(V, batch_size)

        # Step 3: Compute Scaled Dot-Product Attention for each head
        attention_output, attention_weights = sdp_attention(Q, K, V, mask)

        # Step 4: Concatenate the heads
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention_output, (batch_size, -1, self.dm)
        )  # Shape: (batch, seq_len_q, dm)

        # Step 5: Project the concatenated output
        output = self.linear(concat_attention)

        return output, attention_weights
