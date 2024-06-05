"""
Author: Alex Chattos (chattosa1@udayton.edu)

University of Dayton Department of Civil and Environmental Engineering


This module is intended to house the bisection method for finding roots of functions.

Required packages/modules:
   - numpy
   - math
"""
# import the logarithm function from numpy and the floor function from math
from numpy import log10
from math import floor


def bisection_method(func, lower_bound, upper_bound, iterations):
    """
    Finds a root of the function `func` within the interval [lower_bound, upper_bound] using the bisection method.

    Parameters:
    - func: The function for which the root is to be found. It must be a function of a single variable.
    - lower_bound: The lower bound of the interval in which to search for the root.
    - upper_bound: The upper bound of the interval in which to search for the root.
    - iterations: The maximum number of iterations to perform


    Returns:
    - The approximate value of the root if found within the given tolerance and iteration limit,
    - Approximation error
    """
    # Check if the function values at the lower and upper bounds have the same sign
    # If they do, the bisection method fails because it means there's no root in the interval
    if func(lower_bound) * func(upper_bound) >= 0:
        print("Bisection method fails.")
        return None, 0
    # Initialize the interval [a, b] with the given lower and upper bounds
    a = lower_bound
    b = upper_bound
    # Initialize this old variable to store the previous midpoint
    old = 0
    # Perform the bisection method for max_iterations times
    for i in range(iterations):
        # Calculate the midpoint of the interval
        c = (a + b) / 2
        # Calculate the error if it's not the first iteration
        if i != 0:
            error = ((abs(c - old))/c) * 100
        # Save the old midpoint
        old = c
        # Check if root is on [a, c]
        if func(a) * func(c) < 0:
            b = c
        else:
            # Otherwise, the root is in the interval [c, b]
            a = c

    return c, error


def find_sigfigs(error, tolerance):
    """
    Finds the least amount of significant digits correct for the given root and tolerance.

    Parameters:
    - root: The root for which the significant digits are to be found.
    - tolerance: The tolerance within which the root is accepted.

    Returns:
    - The least amount of significant digits correct for the given root and tolerance.

    Method:
    error <= tolerance^(-m) where m is the number of significant digits
    """
    err_over_tol = error / tolerance
    sig_figs = 2 - log10(err_over_tol)
    sig_figs = floor(sig_figs)
    # Return the significant digits
    return sig_figs


if __name__ == "__main__":

    print("You are running this module directly.")
    print("This module is intended to be imported into another program.")
