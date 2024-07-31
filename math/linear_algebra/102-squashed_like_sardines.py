#!/usr/bin/env python3
"""This file defines the cat_matrices function"""


def cat_matrices(mat1, mat2, axis=0):
    """
    This function concatenates two matrices along a specific axis.
    Args:
        mat1 (list): this is the first matrix to concatenate.
        mat2 (list): this is the second matrix to concatenate.
    Returns:
        This function returns a new matrix. If the two matrices cannot be
        concatenated, return None.
    """
    def is_valid_shape(m1, m2, axis):
        """
        Check if two matrices can be concatenated along the specified axis.
        """
        if len(m1) == 0 or len(m2) == 0:
            return False

        # Get shapes of both matrices
        shape1 = [len(m1)]
        shape2 = [len(m2)]

        # Traverse the first matrix to get its complete shape
        tmp = m1
        while isinstance(tmp[0], list):
            shape1.append(len(tmp[0]))
            tmp = tmp[0]

        # Traverse the second matrix to get its complete shape
        tmp = m2
        while isinstance(tmp[0], list):
            shape2.append(len(tmp[0]))
            tmp = tmp[0]

        # Remove the size along the concatenation axis
        shape1.pop(axis)
        shape2.pop(axis)

        # Check if the remaining dimensions are equal
        return shape1 == shape2

    def concatenate(m1, m2, axis):
        """ Concatenate two matrices along the specified axis """
        if axis == 0:
            return m1 + m2
        else:
            return [concatenate(x, y, axis - 1) for x, y in zip(m1, m2)]

    if not is_valid_shape(mat1, mat2, axis):
        return None

    return concatenate(mat1, mat2, axis)
