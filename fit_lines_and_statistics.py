"""
Author: Alex Chattos (chattosa1@udayton.edu)

University of Dayton Department of Civil and Environmental Engineering


This module is intended to house functions for statistical analysis of data to help CEE students interpret formulas encountered during course work

Required packages/modules:
    -numpy
    -matplotlib
    -pandas

    numpy, matplotlib, and pandas should be installed with Spyder IDE installation
    

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pandas as pd
import warnings

# Ignore specific warnings that do not affect the functionality of the code
warnings.filterwarnings("ignore", message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated")
warnings.filterwarnings('ignore', category=np.RankWarning)


""" Place to house functions """

"""
Numpy's polyfit function methodology:
y = a0z0 + a1z1 + a2z2 + ... + a_mz_m + e

where:
    z0 = 1, z1 = x, z2 = x^2, ..., z_m = x^m
    a0, a1, a2, ..., a_m are the coefficients

    [Z] = |z00 z10 z20 ... z_m0|
          |z02 z12 z22 ... z_m2|
          |.    .    .    .    |
          |.    .    .    .    |  n = number of data points
          |.    .    .    .    |
          |z0n z1n z2n ... z_mn|
{y} = [Z]{a} + {e}

[[Z]^T[Z]]{a} = [[Z]^T]{y}
"""


def PolyRegress(x, y, order, plot_flag):
    """
    Perform polynomial regression on the data to the order given by the user.

    Parameters:
    x (numpy array): Independent variable data.
    y (numpy array): Dependent variable data.
    order (int): Order of the polynomial regression.
    plot_flag (bool): Flag to plot the polynomial fit line.

    Returns:
    numpy.ndarray: Fitted polynomial function values.
    string: fit line equation
    float: Coefficient of determination (R^2) of the polynomial fit.
    """
    # Perform polynomial regression
    coeffs = np.polyfit(x, y, order)
    poly_fn = np.poly1d(coeffs)
    pred_poly = poly_fn(x)  # This line generates the pred_poly values
    equation_poly = "y = "
    for i, coeff in enumerate(coeffs[::-1]):
        if i == 0:
            equation_poly += f"{coeff:.4f}"
        elif i == 1:
            equation_poly += f" + {coeff:.4f}x"
        else:
            equation_poly += f" + {coeff:.4f}x^{i}"
    equation_poly = equation_poly.replace("+ -", "- ")
    r_squared_poly = Rsq(pred_poly, y)
    # Plot the polynomial fit
    if plot_flag:
        plt.figure()
        plt.scatter(x, y, color='blue', label='Data')
        x_smooth = np.linspace(x.min(), x.max(), 500)
        y_smooth = np.polyval(coeffs, x_smooth)
        plt.plot(x_smooth, y_smooth, color='cyan', label=f'Polynomial Fit: {equation_poly}, R^2 = {r_squared_poly:.4f}')
        plt.legend()
        plt.show()

    return pred_poly, r_squared_poly, equation_poly




def Rsq(predicted, empirical):
    """
    no numpy function for R^2 so just define it manually :
    
    Calculate the coefficient of determination (R^2) for a regression model.

    Parameters:
    predicted (numpy array): Predicted data from the regression model.
    empirical (numpy array): Empirical data.
    plot_flag (bool): Flag to plot the polynomial fit line.

    Returns:
    float: Coefficient of determination (R^2).
    """
    # Calculate Residual Sum of Squares (RSS) and Total Sum of Squares (TSS)
    RSS = np.sum((empirical - predicted) ** 2)
    TSS = np.sum((empirical - np.mean(empirical)) ** 2)
    if TSS == 0:
        return np.nan  # or return some other value indicating undefined R^2
    else:
        return 1 - (RSS / TSS)


# linear fit function
def linearfit(x, y, plot_flag):
    """
    Fit a linear regression model to the data and optionally calculate linear regression statistics.

    Parameters:
    x (numpy array): Independent variable data.
    y (numpy array): Dependent variable data.
    plot_flag (bool): Flag to plot the linear fit line.

    Returns:
    expected: Expected values of the dependent variable based on the linear fit.
    string: fit line equation
    float: Coefficient of determination (R^2) of the linear fit
    """
    # Fit a linear regression model    
    linear_fit = np.polyfit(x, y, 1)
    linear_fit_fn = np.poly1d(linear_fit)
    linear_fit_fn = linear_fit_fn(x)

    equation_linear = f'y = {linear_fit[0]:.4f}x + {linear_fit[1]:.4f}'
    r_squared_linear = Rsq(linear_fit_fn, y)
    if plot_flag:
        plt.figure()
        plt.scatter(x, y, color='blue', label='Data')
        plt.plot(x, linear_fit_fn, color='red', label=f'Linear Fit: {equation_linear}, R^2 = {r_squared_linear}')
        plt.legend()
        plt.show()
    return linear_fit_fn, r_squared_linear, equation_linear


# Exponential fit function
def exponentialfit(x, y, plot_flag):
    """
    Fit an exponential regression model to the data.
    
    Parameters:
    x (numpy array): Independent variable data.
    y (numpy array): Dependent variable data.
    plot_flag (bool): Flag to plot the exponential fit line.
    
    Returns:
    numpy.ndarray: Fitted exponential function values.
    string: fit line equation
    float: Coefficient of determination (R^2) of the exponential fit.
    """
    # Fit an exponential regression model
    exp_fit = np.polyfit(x, np.log(y), 1, w=np.sqrt(y))
    exp_fit_fn = np.exp(exp_fit[1]) * np.exp(exp_fit[0] * x)
    equation_exp = f'y = {np.exp(exp_fit[1])}e^{exp_fit[0]}x'
    r_squared_exp = Rsq(exp_fit_fn, y)
    if plot_flag:
        plt.figure()
        plt.scatter(x, y, color='blue', label='Data')
        plt.plot(x, exp_fit_fn, color='green', label=f'Exponential Fit: {equation_exp}, R^2 = {r_squared_exp}')
        plt.legend()
        plt.show()
    return exp_fit_fn, r_squared_exp, equation_exp


# Logarithmic fit function
def logfit(x, y, plot_flag):
    """
    Fit a logarithmic regression model to the data and optionally calculate logarithmic regression statistics.

    Parameters:
    x (numpy array): Independent variable data.
    y (numpy array): Dependent variable data.
    plot_flag (bool): Flag to plot the logarithmic fit line.

    Returns:
    numpy.ndarray: Fitted logarithmic function values.
    sting: fit line equation.
    float: Coefficient of determination (R^2) of the logarithmic fit.
    """
    # Fit a logarithmic regression model
    log_fit = np.polyfit(np.log(x), y, 1)
    log_fit_fn = log_fit[0] * np.log(x) + log_fit[1]

    equation_log = f'y = {log_fit[0]}ln(x) + {log_fit[1]}'
    r_squared_log = Rsq(log_fit_fn, y)
    if plot_flag:
        plt.figure()
        plt.scatter(x, y, color='blue', label='Data')
        plt.plot(x, log_fit_fn, color='purple', label=f'Logarithmic Fit: {equation_log}, R^2 = {r_squared_log}')
        plt.legend()
        plt.show()
    return log_fit_fn, r_squared_log, equation_log


# Power fit
def powfit(x, y, plot_flag):
    """
    Fit a power-law regression model to the data and optionally calculate power-law regression statistics.

    Parameters:
    x (numpy array): Independent variable data.
    y (numpy array): Dependent variable data.
    plot_flag (bool): Flag to plot the power-law fit line.

    Returns:
    numpy.ndarray: Fitted power-law function values.
    string: fit line equation
    float: Coefficient of determination (R^2) of the power-law fit.
    """
    # Fit a power-law regression model
    power_fit = np.polyfit(np.log(x), np.log(y), 1)
    power_fit_fn = np.exp(power_fit[1]) * x ** power_fit[0]

    equation_power = f'y = {np.exp(power_fit[1])}x^{power_fit[0]}'
    r_squared_power = Rsq(power_fit_fn, y)
    if plot_flag:
        plt.figure()
        plt.scatter(x, y, color='blue', label='Data')
        plt.plot(x, power_fit_fn, color='orange', label=f'Power Fit: {equation_power}, R^2 = {r_squared_power}')
        plt.legend()
        plt.show()
    return power_fit_fn, r_squared_power, equation_power


# Create a DataFrame to store the fit statistics
def create_dataframe(pred_poly,  linear_fit_fn,  exp_fit_fn, log_fit_fn, power_fit_fn,
                     r_squared_poly, r_squared_linear, r_squared_exp, r_squared_log, r_squared_power,
                     equation_linear, equation_log, equation_power, equation_exp, equation_poly):
    # Initialize an empty DataFrame
    fits_stats = pd.DataFrame(columns=['Fit Type', 'Fitted Function Values', 'R^2', 'Equation'])

    # Polynomial fit
    poly_dict = {'Fit Type': 'Polynomial', 'Fitted Function Values': pred_poly, 'R^2': r_squared_poly, 'Equation': equation_poly}
    fits_stats = fits_stats._append(poly_dict, ignore_index=True)

    # Linear fit

    linear_dict = {'Fit Type': 'Linear', 'Fitted Function Values': linear_fit_fn, 'R^2': r_squared_linear, 'Equation': equation_linear}
    fits_stats = fits_stats._append(linear_dict, ignore_index=True)

    # Exponential fit
    exp_dict = {'Fit Type': 'Exponential', 'Fitted Function Values': exp_fit_fn, 'R^2': r_squared_exp, 'Equation': equation_exp}
    fits_stats = fits_stats._append(exp_dict, ignore_index=True)

    # Logarithmic fit

    log_dict = {'Fit Type': 'Logarithmic', 'Fitted Function Values': log_fit_fn, 'R^2': r_squared_log, 'Equation': equation_log}
    fits_stats = fits_stats._append(log_dict, ignore_index=True)

    # Power fit
    pow_dict = {'Fit Type': 'Power', 'Fitted Function Values': power_fit_fn, 'R^2': r_squared_power, 'Equation': equation_power}
    fits_stats = fits_stats._append(pow_dict, ignore_index=True)

    return fits_stats


def fit_portion(x, y, highest_order):
    """
    This function will split the data into two portions and fit a line to each portion.
    The function will then plot the data and the best fit lines for each portion, based on the maximum R squared sum between the two portions.

    :param x:
    :param y:
    :param highest_order:
    :return: plot of the best fit lines for each portion

    """
    # Initialize the best split and the best R squared sum
    best_split = None
    best_rsq_sum = 0
    best_fit_type1 = None
    best_fit_type2 = None
    best_fit_eqn1 = None
    best_fit_eqn2 = None

    # Loop over all possible split points
    for split_point in range(1, len(x) - 1):
        # Split the data into two portions
        x1, x2 = x[:split_point], x[split_point:]
        y1, y2 = y[:split_point], y[split_point:]

        # Apply the regression functions to each portion
        fit_types = ['PolyRegress', 'linearfit', 'exponentialfit', 'logfit', 'powfit']
        fit_eqns = [f'y = {np.polyfit(x1, y1, highest_order)}',
                    f'y = {np.polyfit(x1, y1, 1)[0]}x + {np.polyfit(x1, y1, 1)[1]}',
                    f'y = {np.exp(np.polyfit(x1, np.log(y1), 1)[1])}e^{np.polyfit(x1, np.log(y1), 1)[0]}x',
                    f'y = {np.polyfit(np.log(x1), y1, 1)[0]}ln(x) + {np.polyfit(np.log(x1), y1, 1)[1]}',
                    f'y = {np.exp(np.polyfit(np.log(x1), np.log(y1), 1)[1])}x^{np.polyfit(np.log(x1), np.log(y1), 1)[0]}']
        rsqs = [Rsq(PolyRegress(x1, y1, highest_order, plot_flag=False)[0], y1),
                Rsq(linearfit(x1, y1, plot_flag=False)[0], y1),
                Rsq(exponentialfit(x1, y1, plot_flag=False)[0], y1),
                Rsq(logfit(x1, y1, plot_flag=False)[0], y1),
                Rsq(powfit(x1, y1, plot_flag=False)[0], y1)]
        max_rsq1 = max(rsqs)
        best_fit_type1 = fit_types[rsqs.index(max_rsq1)]
        best_fit_eqn1 = fit_eqns[rsqs.index(max_rsq1)]

        fit_eqns = [f'y = {np.polyfit(x2, y2, highest_order)}',
                    f'y = {np.polyfit(x2, y2, 1)[0]}x + {np.polyfit(x2, y2, 1)[1]}',
                    f'y = {np.exp(np.polyfit(x2, np.log(y2), 1)[1])}e^{np.polyfit(x2, np.log(y2), 1)[0]}x',
                    f'y = {np.polyfit(np.log(x2), y2, 1)[0]}ln(x) + {np.polyfit(np.log(x2), y2, 1)[1]}',
                    f'y = {np.exp(np.polyfit(np.log(x2), np.log(y2), 1)[1])}x^{np.polyfit(np.log(x2), np.log(y2), 1)[0]}']
        rsqs = [Rsq(PolyRegress(x2, y2, highest_order, plot_flag=False)[0], y2),
                Rsq(linearfit(x2, y2, plot_flag=False)[0], y2),
                Rsq(exponentialfit(x2, y2, plot_flag=False)[0], y2),
                Rsq(logfit(x2, y2, plot_flag=False)[0], y2),
                Rsq(powfit(x2, y2, plot_flag=False)[0], y2)]
        max_rsq2 = max(rsqs)
        best_fit_type2 = fit_types[rsqs.index(max_rsq2)]
        best_fit_eqn2 = fit_eqns[rsqs.index(max_rsq2)]

        # Calculate the sum of the R squared values for the two portions
        rsq_sum = max_rsq1 + max_rsq2

        # If this sum is greater than the best R squared sum, update the best R squared sum and the best split
        if rsq_sum > best_rsq_sum:
            best_rsq_sum = rsq_sum
            best_split = split_point

    # After the loop, best_split is the best split point and best_rsq_sum is the corresponding sum of R squared values
    print(f"Best split point: {best_split}, Best R squared sum: {best_rsq_sum}")

    # Plot the best fit lines for each portion with the equations and R squared values in the legend
    plt.figure()
    plt.scatter(x, y, color='blue', label='Data')
    plt.plot(x[:best_split], eval(f"{best_fit_type1}(x[:best_split], y[:best_split], highest_order, plot_flag=False)[0]"), label=f"{best_fit_type1} Fit 1: {best_fit_eqn1}, R^2 = {max_rsq1}")
    plt.plot(x[best_split:], eval(f"{best_fit_type2}(x[best_split:], y[best_split:], highest_order, plot_flag=False)[0]"), label=f"{best_fit_type2} Fit 2: {best_fit_eqn2}, R^2 = {max_rsq2}")
    plt.legend()
    plt.show()


# Main function to test the functions (aka if you run this script it will run this function)
if __name__ == '__main__':
    print("This is the main function, it will not run unless you run this script directly. ")
