#!/usr/bin/env python3
""" This module defines the class EncoderBlock for a transformer. """
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """ This class represents an encoder block for a transformer. """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        This method initializes the Transformer Encoder Block.
        Args:
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
            hidden (int): Number of hidden units in the feed forward layer.
            drop_rate (float): Dropout rate.
        """
        super(EncoderBlock, self).__init__()

        # Multi-Head Attention
        self.mha = MultiHeadAttention(dm, h)

        # Dense layers for the feed-forward network
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation="relu")
        self.dense_output = tf.keras.layers.Dense(dm)

        # Layer Normalization
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout layers
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        This method executes the Encoder Block.
        Args:
            x (tf.Tensor): Input tensor of shape (batch, seq_len, dm).
            training (bool): Whether the model is in training mode.
            mask (tf.Tensor): Mask to apply during attention. Defaults to None.
        Returns:
            tf.Tensor: Output tensor of shape (batch, seq_len, dm).
        """
        # Multi-Head Attention sublayer
        attn_output, _ = self.mha(x, x, x, mask)  # Self-attention
        attn_output = self.dropout1(attn_output, training=training)
        # Residual connection + LayerNorm
        out1 = self.layernorm1(x + attn_output)

        # Feed Forward Network sublayer
        ffn_output = self.dense_hidden(out1)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        # Residual connection + LayerNorm
        out2 = self.layernorm2(out1 + ffn_output)

        return out2
