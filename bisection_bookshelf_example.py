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
Lboundary = 0
Rboundary = 29

# find the root using the bisection method
root, sig_figs = BM.bisection_method(bookshelf_function, Lboundary, Rboundary, tolerance=1e-4, max_iterations=10)
print(f"Found root: {root} with {sig_figs} significant figures.")
