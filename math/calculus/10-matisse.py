#!/usr/bin/env python3
"""This file defines the poly_derivative function."""


def poly_derivative(poly):
    """
    This function calculates the derivative of a polynomial.
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    n = len(poly)
    if n == 1:
        return [0]
    poly_deriv = []
    for k in range(1, n):
        new_coef = k * poly[k]
        poly_deriv.append(new_coef)
    return poly_deriv
