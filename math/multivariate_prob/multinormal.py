#!/usr/bin/env python3
""" This module defines the MultiNormal class. """
import numpy as np


class MultiNormal:
    """ This class represents a Multivariate Normal distribution. """

    def __init__(self, data):
        """
        This method initializes the MultiNormal object.
        Args:
            data (numpy.ndarray): Is an array of shape (d, n) containing the
                data set.
                - d is the number of dimensions.
                - n is the number of data points.
        Returns:
            None.
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        n = data.shape[1]
        self.mean = np.mean(data, axis=1, keepdims=True)
        self.cov = np.dot(data - self.mean, (data - self.mean).T) / (n - 1)
