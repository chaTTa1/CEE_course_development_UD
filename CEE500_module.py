"""
Author: Alex Chattos (chattosa1@udayton.edu)

University of Dayton Department of Civil and Environmental Engineering


This module is intended to house functions to help CEE students interpret formulas encountered during course work

Required packages/modules:
    -numpy
    -matplotlib
    -pandas

    numpy, matplotlib, and pandas should be installed with Spyder IDE installation
"""


# Importing necessary packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

import numpy as np
import re

def solve_user_equations_with_knowns(num_equations):
    coefficients = []
    constants = []
    variables = []

    # Prompt for known variables and their values
    knowns_input = input("Enter known variables and their values (e.g., 'x=2 y=3'), or press enter if none: ")
    known_variables = dict(re.findall(r'(\w+)=(\d+)', knowns_input))

    # Convert known variable values to floats
    for var in known_variables:
        known_variables[var] = float(known_variables[var])

    # Prompt for variables
    variables_input = input("Enter the variables separated by space (e.g., 'x y'): ")
    variables = variables_input.split()

    # Remove known variables from the list of variables to solve for
    variables = [var for var in variables if var not in known_variables.keys()]

    # Loop through the number of equations
    for _ in range(num_equations):
        equation = input("Enter an equation (e.g., '3x + 2y = 5'): ")

        # Replace known variables with their values in the equation
        for var, value in known_variables.items():
            equation = equation.replace(var, str(value))

        # Find all numbers in the equation
        numbers = re.findall(r'[-+]?\d*\.?\d+|[-+]?\d+', equation)

        # Assuming the last number is the constant (right side of the equation)
        constant = numbers.pop()

        # Convert strings to floats
        numbers = [float(num) for num in numbers]
        constant = float(constant)

        # Append the coefficients and constant to their respective arrays
        coefficients.append(numbers)
        constants.append(constant)

    # Convert lists to numpy arrays
    coefficients = np.array(coefficients)
    constants = np.array(constants)

    # Solve the system of equations
    solution = np.linalg.solve(coefficients, constants)

    # Map solutions to variables
    solution_dict = {var: sol for var, sol in zip(variables, solution)}

    # Add known variables to the solution dictionary
    solution_dict.update(known_variables)

    return solution_dict

