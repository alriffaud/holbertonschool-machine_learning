#!/usr/bin/env python3
""" This module defines the sdp_attention function for a transformer. """
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    This function computes the scaled dot product attention.
    Args:
        Q (tf.Tensor): Query matrix, shape (..., seq_len_q, dk).
        K (tf.Tensor): Key matrix, shape (..., seq_len_v, dk).
        V (tf.Tensor): Value matrix, shape (..., seq_len_v, dv).
        mask (tf.Tensor, optional): Mask tensor, broadcastable to
                                     (..., seq_len_q, seq_len_v).
    Returns:
        output (tf.Tensor): Scaled dot product attention output,
                            shape (..., seq_len_q, dv).
        weights (tf.Tensor): Attention weights,
                             shape (..., seq_len_q, seq_len_v).
    """
    # Step 1: Compute QKᵀ (similarity scores)
    # Shape: (..., seq_len_q, seq_len_v)
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # Step 2: Scale by sqrt(dk)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)  # Dimension of keys
    scaled_scores = matmul_qk / tf.math.sqrt(dk)

    # Step 3: Apply mask (if provided)
    if mask is not None:
        scaled_scores += (mask * -1e9)

    # Step 4: Compute softmax to get attention weights
    # Shape: (..., seq_len_q, seq_len_v)
    weights = tf.nn.softmax(scaled_scores, axis=-1)

    # Step 5: Multiply weights by V to get the output
    output = tf.matmul(weights, V)  # Shape: (..., seq_len_q, dv)

    return output, weights
