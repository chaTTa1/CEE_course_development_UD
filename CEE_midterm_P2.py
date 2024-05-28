"""
Author: Alex Chattos (chattosa1@udayton.edu)

University of Dayton Department of Civil and Environmental Engineering

This script is used to solve the second problem of the CEE 500 midterm exam.

Required packages/modules:
    - sympy
    - CEE_500_module (module created for solving the system of equations and plotting the results)
    sympy should be installed with Spyder IDE installation
"""
# import necessary packages
import sympy as sp
import CEE_500_module as cee

# Define symbols, these are the variables that will be solved for
# sympy.symbols is used to define symbolic variables
thetaC, thetaE, delta, Sac, Sbe = sp.symbols('thetaC thetaE delta Sac Sbe')

# Given values
EIb = 3418.05
EIc = 1277.77

# Define moments, these are the moment equations
# The moments are defined as expressions involving the symbolic variables
# These expressions will be used in the equations to be solved
moments = {
    "Mac": (2/24) * EIc * (thetaC + (delta / 8)),
    "Mca": (2/24) * EIc * (2 * thetaC + (delta / 8)),
    "Mce": (2/14) * EIb * (2 * thetaC + thetaE) - 14.43,
    "Mec": (2/14) * EIb * (2 * thetaE + thetaC) + 22.3,
    "Meb": (2/24) * EIc * (2 * thetaE + (delta / 8)),
    "Mbe": (2/24) * EIc * (thetaE + (delta / 8))
}

# Define constants
constants = (thetaC, delta, thetaE, Sac, Sbe)

# Define equations
# The equations are defined using sympy.Eq, which represents an equation
# The left-hand side of the equation is the sum of moments, and the right-hand side is a constant
# These equations will be solved for the symbolic variables
equations = [
    sp.Eq(moments["Mca"] + moments["Mce"], 0),
    sp.Eq(moments["Meb"] + moments["Mec"] - 16.5, 0),
    sp.Eq(10 - Sac - Sbe, 0),
    sp.Eq(moments["Mac"] + moments["Mca"] - Sac * 24, 0),
    sp.Eq(moments["Mbe"] + moments["Meb"] - Sbe * 24, 0)
]

# Call the function
# The solve_equations function from the CEE_500_module is used to solve the equations
# The solution is a dictionary where the keys are the symbolic variables and the values are their solved values
solution, moments_val = cee.solve_equations(equations, moments, constants)

# Print the solution and the moments
print(solution)
print(moments_val)

# plot the moments
# The plot_moments function from the CEE_500_module is used to plot the moments
cee.plot_moments(moments_val)