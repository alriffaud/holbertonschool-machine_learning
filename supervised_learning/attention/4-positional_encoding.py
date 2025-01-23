#!/usr/bin/env python3
""" This module defines the positional_encoding function for a transformer. """
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    This function calculates the positional encoding for a transformer.
    Args:
        max_seq_len (int): represents the maximum sequence length.
        dm (int): is the model depth.
    Returns:
        np.ndarray: Array of shape (max_seq_len, dm) containing the positional
            encoding vectors.
    """
    # Initialize the positional encoding matrix
    PE = np.zeros((max_seq_len, dm))

    # Compute the positional encoding
    for pos in range(max_seq_len):
        for j in range(0, dm, 2):
            angle = pos / 10000 ** (j / dm)
            PE[pos, j] = np.sin(angle)
            PE[pos, j + 1] = np.cos(angle)

    return PE
