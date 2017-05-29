import numpy as np
from numpy import pi



def erf(x, method='transcendental'):
    """
    erf(x[, method])
    computes an approximation to the error function.

    This is designed to be very accurate in a neighborhood of 0 and a
    neighborhood of infinity, and the error is less than 0.00035 for all
    x. Using the alternate value a ≈ 0.147 reduces the maximum error to about
    0.00012.

    Winitzki, Sergei (6 February 2008). "A handy approximation for the error
    function and its inverse" (PDF). Retrieved 2011-10-03.
    """
    if x < 0:
        return -erf(-x, method)
    if method is 'poly_low':
        return
    else:
        a = 8 * (pi - 3) / (3 * pi * (4-pi))  # = .140012...
        exp_arg = - x**2 * (4/pi + a * x**2)/(1 + a * x**2)
        return np.sign(x) * np.sqrt(1 - np.exp(exp_arg))

def erfinv(x):
    """
    erfinv(x)
    computes an approximation to the inverse of the error function.

    This is designed to be very accurate in a neighborhood of 0 and a
    neighborhood of infinity, and the error is less than 0.00035 for all
    x. Using the alternate value a ≈ 0.147 reduces the maximum error to about
    0.00012.

    Winitzki, Sergei (6 February 2008). "A handy approximation for the error
    function and its inverse" (PDF). Retrieved 2011-10-03.
    """
    a = 8 * (pi - 3) / (3 * pi * (4-pi))  # = .140012...
    b = 2 / (pi * a)
    c = np.log(1 - x**2)/2
    A = (b + c)**2
    B = 2*c/a
    C = b + c
    return np.sign(x) * np.sqrt(np.sqrt(A - B) - C)
