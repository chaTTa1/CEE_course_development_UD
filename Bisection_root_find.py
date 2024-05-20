"""
Author: Alex Chattos (chattosa1@udayton.edu)

University of Dayton Department of Civil and Environmental Engineering


This module is intended to house the bisection method for finding roots of functions.

Required packages/modules:
    -none
"""


def bisection_method(func, lower_bound, upper_bound, tolerance, max_iterations):
    """
    Finds a root of the function `func` within the interval [lower_bound, upper_bound] using the bisection method.
    Also calculates the least amount of significant digits correct for the found root.

    Parameters:
    - func: The function for which the root is to be found. It must be a function of a single variable.
    - lower_bound: The lower bound of the interval in which to search for the root.
    - upper_bound: The upper bound of the interval in which to search for the root.
    - tolerance: The tolerance within which the root is accepted. Default is 1e-5.
    - max_iterations: The maximum number of iterations to perform. Default is 1000.

    Returns:
    - The approximate value of the root if found within the given tolerance and iteration limit, otherwise None.
    - The least amount of significant digits correct for the found root.
    """

    if func(lower_bound) * func(upper_bound) >= 0:
        print("Bisection method fails.")
        return None, 0

    a = lower_bound
    b = upper_bound
    for _ in range(max_iterations):
        c = (a + b) / 2
        if abs(func(c)) < tolerance:
            # Calculate the least amount of significant digits correct
            sig_digits = 0
            test_tolerance = 1e-1  # Start with a tolerance of 0.1
            while abs(func(c)) < test_tolerance and sig_digits <= 15:  # Limit to 15 significant digits
                sig_digits += 1
                test_tolerance /= 10  # Decrease tolerance by an order of magnitude
            return c, sig_digits
        if func(a) * func(c) < 0:
            b = c
        else:
            a = c

    print("Max iterations reached. Approximate solution or divergence.")
    return None, 0