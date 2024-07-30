#!/usr/bin/env python3
"""This file defines the np_matmul function"""
import numpy as np


def np_matmul(mat1, mat2):
    """
    This function performs matrix multiplication.
    Args:
        mat1 (list): this is the first matrix.
        mat2 (list): this is the second matrix.
    Returns:
        This function returns a matrix representing the multiplication of the
        two matrices.
    """
    return mat1 @ mat2
