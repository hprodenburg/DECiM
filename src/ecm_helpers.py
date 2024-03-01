"""Part of DECiM. This file contains small helper functions. Last modified 6 December 2023 by Henrik Rodenburg.

Functions:
nearest -- find index of element in a list whose value is closest to a given value
maxima -- find the indices of all elements in a list whose values are higher than those of the surrounding elements"""

###########
##IMPORTS##
###########

import numpy as np

####################
##HELPER FUNCTIONS##
####################

def nearest(a, b):
    """For a scalar value a, find the closest value in list b and return its index.
    
    Arguments:
    a -- float
    b -- list
    
    Returns:
    Index of element in b whose value is closest to a, assuming that the value of b increases or decreases throughout the length of the list"""
    c = list((np.array(b) - a)**2)
    return c.index(min(c))

def maxima(a):
    """Find every value in list a which is higher than the previous value and higher than the next value in the list. Return a list of all these values' indices.
    
    Arguments:
    a -- list of scalar values
    
    Returns:
    list of indices of all elements whose value is higher than that of the elements at the two surrounding indices."""
    res = []
    for i in range(len(a)):
        if i > 0 and i < len(a) - 1:
            if a[i] > a[i - 1] and a[i] > a[i + 1]:
                res.append(i)
    return res
