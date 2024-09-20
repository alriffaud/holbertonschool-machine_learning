#!/usr/bin/env python3
""" This module defines the early_stopping function. """
import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    This function determines if you should stop gradient descent early.
    Args:
        cost (float): The current validation cost of the network.
        opt_cost (float): The lowest recorded validation cost of the network.
        threshold (float): The threshold used for early stopping.
        patience (int): The patience count used for early stopping.
        count (int): The count of how long the threshold has not been met.
    Returns: a boolean of whether the network should be stopped early, followed
    by the updated count.
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    if count >= patience:
        return True, count
    return False, count
