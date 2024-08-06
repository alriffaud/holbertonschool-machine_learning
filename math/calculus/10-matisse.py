#!/usr/bin/env python3
"""This file defines the poly_derivative function."""


def poly_derivative(poly):
    """
    This function calculates the derivative of a polynomial.
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    poly_deriv = []
    n = len(poly) - 1
    for k in range(n, 0, -1):
        new_coef = k * poly[k]
        poly_deriv.append(new_coef)
    return poly_deriv
