#!/usr/bin/env python3
""" This module defines the class Encoder for a transformer. """
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """This class creates the encoder for a transformer"""
    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """
        This method initializes the Encoder.
        Args:
        - N (int): Number of encoder blocks.
        - dm (int): Dimensionality of the model.
        - h (int): Number of attention heads.
        - hidden (int): Hidden units in the feed-forward layers.
        - input_vocab (int): Size of the input vocabulary.
        - max_seq_len (int): Maximum sequence length.
        - drop_rate (float): Dropout rate.
        """
        super(Encoder, self).__init__()
        self.dm = dm
        self.N = N

        # Embedding layer for converting tokens to dense vectors
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)

        # Precompute positional encodings
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        # Encoder blocks (stacked N times)
        self.blocks = [EncoderBlock(dm, h, hidden,
                                    drop_rate) for _ in range(N)]

        # Dropout applied to the embeddings and positional encodings
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        This method performs the forward pass for the Encoder.
        Args:
        - x (Tensor): Input tensor of shape (batch, input_seq_len).
        - training (bool): Whether the model is training.
        - mask (Tensor): Mask to be applied for multi-head attention.
        Returns:
        - Tensor of shape (batch, input_seq_len, dm): Encoded representation.
        """
        # Apply embedding layer
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)  # (batch, input_seq_len, dm)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))  # Scale embeddings

        # Add positional encodings to the embeddings
        x += self.positional_encoding[:seq_len]

        # Apply dropout
        x = self.dropout(x, training=training)

        # Pass through each encoder block
        for block in self.blocks:
            x = block(x, training=training, mask=mask)

        return x
