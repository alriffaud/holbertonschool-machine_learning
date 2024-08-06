#!/usr/bin/env python3
"""This file defines the poly_integral function."""


def poly_integral(poly, C=0):
    """
    This function calculates the derivative of a polynomial.
    """
    if not isinstance(poly, list) or len(poly) == 0 or not isinstance(C, int):
        return None
    n = len(poly)
    poly_int = [C]
    if poly == [0]:
        return poly_int
    for k in range(n):
        new_coef = poly[k] / (k + 1)
        if new_coef.is_integer():
            new_coef = int(new_coef)
        poly_int.append(new_coef)
    return poly_int
