#!/usr/bin/env python3
""" This module defines the class DecoderBlock for a transformer. """
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """ This class represents a decoder block for a transformer. """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        This method initializes the Transformer Decoder Block.
        Args:
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
            hidden (int): Number of hidden units in the FFN.
            drop_rate (float): Dropout rate.
        """
        super(DecoderBlock, self).__init__()

        # Multi-Head Attention layers
        self.mha1 = MultiHeadAttention(dm, h)  # Masked self-attention
        self.mha2 = MultiHeadAttention(dm, h)  # Encoder-decoder attention

        # Feed Forward Network (FFN)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation="relu")
        self.dense_output = tf.keras.layers.Dense(dm)

        # Normalization layers
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout layers
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        This method executes the Decoder Block.
        Args:
            x (tf.Tensor): Input tensor of shape (batch, target_seq_len, dm).
            encoder_output (tf.Tensor): Output from the encoder (batch,
                input_seq_len, dm).
            training (bool): Training mode flag.
            look_ahead_mask (tf.Tensor): Mask for the first attention layer.
            padding_mask (tf.Tensor): Mask for the second attention layer.
        Returns:
            tf.Tensor: Output tensor of shape (batch, target_seq_len, dm).
        """
        # Step 1: Masked Self-Attention
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)  # Residual connection + LayerNorm

        # Step 2: Encoder-Decoder Attention
        attn2, _ = self.mha2(out1, encoder_output,
                             encoder_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)  # Residual connection + LayerNorm

        # Step 3: Feed Forward Network
        ffn_output = self.dense_hidden(out2)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        # Residual connection + LayerNorm
        out3 = self.layernorm3(out2 + ffn_output)

        return out3
