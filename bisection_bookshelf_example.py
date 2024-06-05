"""
Author: Alex Chattos (chattosa1@udayton.edu)

University of Dayton Department of Civil and Environmental Engineering


This file takes the bisection method and applies it to the bookshelf function from the CEE500_module.py file.

Required packages/modules:
    -Bisection_root_find
"""
# import the bisection method function
import Bisection_root_find as BM


# define the function
def bookshelf_function(x):
    return -0.67665e-8 * x ** 4 - 0.26689e-5 * x ** 3 + 0.12748e-3 * x ** 2 - 0.018507


# define boundaries
Left_boundary = 0
Right_boundary = 29
# Find the root of the function and the error, then calculate the number of significant figures correct
root, error = BM.bisection_method(bookshelf_function, Left_boundary, Right_boundary, 10)
print(f"Error: {error}")
sig_figs = BM.find_sigfigs(error, 0.5)
print(f"Found root: {root} with at least {sig_figs} significant figures correct.")
