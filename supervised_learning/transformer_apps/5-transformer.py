#!/usr/bin/env python3
""" This module defines the class Transformer for a transformer network. """
import tensorflow as tf
import sys
sys.path.append('../attention')
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """ This class defines the Transformer model."""
    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        This method initializes the Transformer model.
        Args:
            N: Number of blocks in the Encoder and Decoder.
            dm: Dimensionality of the model.
            h: Number of heads.
            hidden: Number of hidden units in the FFN.
            input_vocab: Size of the input vocabulary.
            target_vocab: Size of the target vocabulary.
            max_seq_input: Maximum sequence length possible for input.
            max_seq_target: Maximum sequence length possible for target.
            drop_rate: Dropout rate.
        """
        super(Transformer, self).__init__()

        # Initialize Encoder with N blocks
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)

        # Initialize Decoder with N blocks
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)

        # Final dense layer to project output to target vocabulary
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """
        This method performs the forward pass for the Transformer model.
        Args:
            inputs: Tensor of shape (batch, input_seq_len) - input sequences.
            target: Tensor of shape (batch, target_seq_len) - target sequences.
            training: Boolean indicating training mode.
            encoder_mask: Padding mask for the encoder.
            look_ahead_mask: Look-ahead mask for the decoder.
            decoder_mask: Padding mask for the decoder.
        Returns:
            Tensor of shape (batch, target_seq_len, target_vocab).
        """
        # Encoder forward pass
        encoder_output = self.encoder(inputs, training, encoder_mask)

        # Decoder forward pass
        decoder_output = self.decoder(target, encoder_output, training,
                                      look_ahead_mask, decoder_mask)

        # Project decoder output to target vocabulary space
        output = self.linear(decoder_output)

        return output
