#!/usr/bin/env python3
""" This module defines the class Decoder for a transformer. """
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """ This class defines the Decoder for a Transformer model. """
    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1):
        """
        This method initializes the Decoder.
        Args:
            N: Number of Decoder blocks.
            dm: Dimensionality of the model.
            h: Number of attention heads.
            hidden: Number of hidden units in the FFNN.
            target_vocab: Size of the target vocabulary.
            max_seq_len: Maximum sequence length.
            drop_rate: Dropout rate.
        """
        super(Decoder, self).__init__()

        # Save model dimensionality and number of blocks
        self.dm = dm
        self.N = N

        # Embedding layer for target tokens
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)

        # Precomputed positional encodings
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        # List of N DecoderBlocks
        self.blocks = [DecoderBlock(dm, h, hidden,
                                    drop_rate) for _ in range(N)]

        # Dropout layer
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        This method performs the forward pass for the Decoder.
        Args:
            x: Tensor of shape (batch, target_seq_len) containing the input to
                the decoder.
            encoder_output: Tensor of shape (batch, input_seq_len, dm)
                containing the encoder's output.
            training: Boolean indicating if the model is in training mode.
            look_ahead_mask: Mask for the first attention layer (look-ahead).
            padding_mask: Mask for the second attention layer (cross-attention)
        Returns:
            Tensor of shape (batch, target_seq_len, dm).
        """
        seq_len = tf.shape(x)[1]

        # Convert tokens to embeddings
        x = self.embedding(x)  # Shape: (batch, target_seq_len, dm)

        # Scale embeddings by sqrt(dm) for stability
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # Add positional encodings
        x += self.positional_encoding[:seq_len]

        # Apply dropout
        x = self.dropout(x, training=training)

        # Pass through each DecoderBlock
        for block in self.blocks:
            x = block(x, encoder_output, training=training,
                      look_ahead_mask=look_ahead_mask,
                      padding_mask=padding_mask)

        return x
