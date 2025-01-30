#!/usr/bin/env python3
"""
This script defines a function that creates masks for Transformer training.
"""
import tensorflow as tf


def create_masks(inputs, target):
    """
    This function creates masks for Transformer training:
    Args:
        inputs: tf.Tensor of shape (batch_size, seq_len_in), input sentence.
        target: tf.Tensor of shape (batch_size, seq_len_out), target sentence.
    Returns:
        encoder_mask: Padding mask for encoder, shape
            (batch_size, 1, 1, seq_len_in)
        combined_mask: Lookahead + padding mask for decoder self-attention,
            shape (batch_size, 1, seq_len_out, seq_len_out)
        decoder_mask: Padding mask for decoder encoder-decoder attention,
            shape (batch_size, 1, 1, seq_len_in)
    """

    # 1️ - Encoder Mask (Padding Mask)
    # Checks where inputs are 0 (padding tokens) and creates a mask
    # (1 where padding exists, 0 otherwise)
    encoder_mask = tf.cast(
        tf.math.equal(inputs, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

    # 2️ - Decoder Mask (Padding Mask for Encoder-Decoder Attention)
    # Same as encoder mask, but applied to decoder's attention over
    # encoder output
    decoder_mask = tf.cast(
        tf.math.equal(inputs, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

    # 3️ - Lookahead Mask (for future words blocking)
    seq_len_out = target.shape[1]
    lookahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len_out, seq_len_out)), -1, 0)

    # 4️ - Target Padding Mask
    target_padding_mask = tf.cast(
        tf.math.equal(target, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

    # 5️ - Combined Mask (Lookahead Mask + Target Padding Mask)
    combined_mask = tf.maximum(target_padding_mask, lookahead_mask)

    return encoder_mask, combined_mask, decoder_mask
