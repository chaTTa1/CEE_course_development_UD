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
    # Check if the function values at the lower and upper bounds have the same sign
    # If they do, the bisection method fails because it means there's no root in the interval
    if func(lower_bound) * func(upper_bound) >= 0:
        print("Bisection method fails.")
        return None, 0
    # Initialize the interval [a, b] with the given lower and upper bounds
    a = lower_bound
    b = upper_bound

    # Perform the bisection method for max_iterations times
    for _ in range(max_iterations):
        # Calculate the midpoint of the interval
        c = (a + b) / 2

        # If the function value at c is less than the tolerance, c is a root
        if abs(func(c)) < tolerance:
            # Calculate the least amount of significant digits correct
            sig_digits = 0
            test_tolerance = 1e-1  # Start with a tolerance of 0.1

            # Increase the number of significant digits as long as the function value at c is less than the test tolerance
            while abs(func(c)) < test_tolerance and sig_digits <= 15:  # Limit to 15 significant digits
                sig_digits += 1
                test_tolerance /= 10  # Decrease tolerance by an order of magnitude

            # Return the root and the number of significant digits
            return c, sig_digits
        # If the function values at a and c have different signs, the root is in the interval [a, c]
        if func(a) * func(c) < 0:
            b = c
        else:
            # Otherwise, the root is in the interval [c, b]
            a = c

    # If the maximum number of iterations is reached, return None
    print("Max iterations reached. Approximate solution or divergence.")
    return None, 0


if __name__ == "__main__":

    print("You are running this module directly.")
    print("This module is intended to be imported into another program.")