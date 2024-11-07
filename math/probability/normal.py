#!/usr/bin/env python3
""" This module contains the Normal class. """


class Normal:
    """ This class represents a normal distribution """
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Constructor for the Normal class.
        Args:
            data (List): Is a list of the data to be used to estimate the
                distribution. Defaults to None.
            mean (float): Is the mean of the distribution. Defaults to 0.
            stddev (float): The standard deviation of the distribution.
                            Defaults to 1.
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            self.stddev = (sum([(x - self.mean) ** 2 for x in data])
                           / len(data)) ** 0.5

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value.
        Args:
            x (float): The x-value.
        Returns:
            The z-score of x.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score.
        Args:
            z (float): The z-score.
        Returns:
            The x-value of z.
        """
        return z * self.stddev + self.mean
