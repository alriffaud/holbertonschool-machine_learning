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

    def pdf(self, x):
        """
        This function calculates the PDF at a data point.
        Args:
            x (numpy.ndarray): Is an array of shape (d, 1) containing the
                data point whose PDF should be calculated. d is the number
                of dimensions.
        Returns:
            The value of the PDF.
        """
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")
        # Calculate the difference between x and the mean
        diff = x - self.mean
        # Calculate the determinant and inverse of the covariance matrix
        cov_det = np.linalg.det(self.cov)
        cov_inv = np.linalg.inv(self.cov)
        # Calculate the denominator
        denom = np.sqrt(((2 * np.pi) ** d) * cov_det)
        # Calculate the exponent
        exponent = -0.5 * (diff.T @ cov_inv @ diff)
        # Calculate the PDF value
        pdf_value = (1 / denom) * np.exp(exponent)
        return pdf_value.item()
