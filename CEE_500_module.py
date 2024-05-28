"""
Author: Alex Chattos (chattosa1@udayton.edu)

University of Dayton Department of Civil and Environmental Engineering


This module is intended to house functions to help CEE 500 students solve a system of equations and plot the results.

Required packages/modules:
    -sympy
    -matplotlib

    matplotlib, and sympy should be installed with Spyder IDE installation
"""
import sympy as sp
import matplotlib.pyplot as plt


def solve_equations(equations, moments, constants):
    # Solve the system of equations
    # sympy.solve is used to solve the equations for the symbolic variables
    # The solution is a dictionary where the keys are the symbolic variables and the values are their solved values
    solution = sp.solve(equations, constants)

    # Substitute the solution into the moments
    # The solved values of the symbolic variables are substituted into the moments
    # The result is a dictionary where the keys are the moments and the values are their calculated values
    moments_val = {moment: expression.subs(solution) for moment, expression in moments.items()}

    # Return the solution and the moments
    return solution, moments_val


def plot_moments(moments):
    # Create a list of moment names and their corresponding values
    moment_names = list(moments.keys())
    moment_values = [float(moment.evalf()) for moment in moments.values()]

    # Create a bar chart
    # matplotlib.pyplot.bar is used to create a bar chart
    # The x-axis represents the different moments and the y-axis represents their values
    plt.bar(moment_names, moment_values)

    # Add labels and title
    plt.xlabel('Moments')
    plt.ylabel('Values')
    plt.title('Moment Diagram')

    # Display the plot
    # matplotlib.pyplot.show is used to display the plot
    plt.show()
    return


if __name__ == "__main__":
    print("You are running this module directly.")
    print("This module is intended to be imported into another program.")