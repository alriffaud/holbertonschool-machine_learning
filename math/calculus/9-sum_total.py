#!/usr/bin/env python3
"""This file defines the summation_i_squared function."""


def summation_i_squared(n):
    """
    This function that calculates the sum of i^2 from 1 to n.
    """
    if type(n) != int:
        return None
    result = n * (n + 1) * (2 * n + 1) // 6
    return result
