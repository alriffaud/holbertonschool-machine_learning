#!/usr/bin/env python3
""" This module defines the moving_average function. """
import numpy as np


def moving_average(data, beta):
    """
    This function calculates the weighted moving average of a data set using
    bias correction.
    Args:
        data (numpy.ndarray):  is the list of data to calculate the moving
        average of.
        beta (float): is the weight used for the moving average.
    Returns: a list containing the moving averages of data.
    """
    v = 0
    moving_averages = []
    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]
        moving_averages.append(v / (1 - beta ** (i + 1)))
    return moving_averages
