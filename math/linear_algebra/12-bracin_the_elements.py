#!/usr/bin/env python3
"""This file defines the np_elementwise function"""


def np_elementwise(mat1, mat2):
    """
    This function performs element-wise addition, subtraction, multiplication,
    and division.
    Args:
        mat1 (list): this is the first matrix.
        mat2 (list): this is the second matrix.
    Returns:
        This function returns a tuple containing the element-wise sum,
        difference, product, and quotient, respectively.
    """
    sum = mat1 + mat2
    dif = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return (sum, dif, mul, div)
