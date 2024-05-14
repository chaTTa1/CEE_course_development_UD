"""
Author: Alex Chattos (chattosa1@udayton.edu)

University of Dayton Department of Civil and Environmental Engineering


This module is intended to house functions for statistical analysis of data to help CEE students interpret formulas encountered during course work

Required packages/modules:
    -numpy
    -matplotlib
    -pandas

    numpy, matplotlib, and pandas should be installed with Spyder IDE installation
    
Optional packages:
    -scipy (for linear regression functions)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pandas as pd
import warnings

warnings.filterwarnings('ignore', category=np.RankWarning)
""" Place to house functions """


def PolyRegres(x, y, order, plot_flag):
    """
    Perform polynomial regression on the data to the order given by the user.

    Parameters:
    x (numpy array): Independent variable data.
    y (numpy array): Dependent variable data.
    order (int): Order of the polynomial regression.
    plot_flag (bool): Flag to plot the polynomial fit line.

    Returns:
    numpy.ndarray: Fitted polynomial function values.
    """
    # Perform polynomial regression


    """
    Methodology: Generalized Regression
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
    # Create the Z matrix from Vandermonde matrix using numpy's vander function
    Z = np.vander(x.flatten(), N=order + 1, increasing=True)

    # Solve for the coefficients
    # Assuming Z and y are defined
    Z_transpose = Z.T

    # Calculate Z'Z
    Z_transpose_Z = np.dot(Z_transpose, Z)

    # Add a small value to the diagonal elements of the matrix
    Z_transpose_Z = Z_transpose_Z.astype(float)
    Z_transpose_Z += np.eye(Z_transpose_Z.shape[0]) * 1e-5

    # Calculate Z'y
    Z_transpose_y = np.dot(Z_transpose, y)

    # Solve for a using numpy's linear algebra solve function
    a = np.linalg.solve(Z_transpose_Z, Z_transpose_y)
    # Calculate the predicted values
    pred_poly = Z @ a

    # Plot the polynomial fit
    if plot_flag:
        plt.plot(x, pred_poly, color='cyan', label='Polynomial Fit')

    # Calculate the coefficient of determination (R^2)
    r_squared_poly = Rsq(pred_poly, y)
    return pred_poly, r_squared_poly




def Rsq(predicted, empirical):
    """
    no numpy function for R^2 so just define it manually :
    
    Calculate the coefficient of determination (R^2) for a regression model.

    Parameters:
    predicted (numpy array): Predicted data from the regression model.
    empirical (numpy array): Empirical data.
    
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
    linreg_flag (bool): Flag to calculate linear regression statistics using linregress from scipy.
    plot_flag (bool): Flag to plot the linear fit line.

    Returns:
    numpy.poly1d: Linear fit function.
    float: Slope of the linear fit line (if linreg_flag is True).
    float: Intercept of the linear fit line (if linreg_flag is True).
    float: Coefficient of determination (R^2) of the linear fit (if linreg_flag is True).
    """
    # Fit a linear regression model    
    linear_fit, intercept_linear = np.polyfit(x, y, 1)
    linear_fit_fn = np.poly1d(linear_fit)
    linear_fit_fn = linear_fit_fn(x)

    equation_linear = f'y = {linear_fit}x + {intercept_linear}'
    r_squared_linear = Rsq(linear_fit_fn, y)
    if plot_flag:
        plt.figure()
        plt.scatter(x, y, color='blue', label='Data')
        plt.plot(x, linear_fit_fn, color='red', label=f'Linear Fit: {equation_linear}, R^2 = {r_squared_linear}')
        plt.legend()
        plt.show()
    return linear_fit_fn, r_squared_linear


# Exponential fit function
def exponentialfit(x, y, plot_flag):
    """
    Fit an exponential regression model to the data.
    
    Parameters:
    x (numpy array): Independent variable data.
    y (numpy array): Dependent variable data.
    
    Returns:
    numpy.ndarray: Fitted exponential function values.
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
    return exp_fit_fn, r_squared_exp


# Logarithmic fit function
def logfit(x, y, linreg_flag, plot_flag):
    """
    Fit a logarithmic regression model to the data and optionally calculate logarithmic regression statistics.

    Parameters:
    x (numpy array): Independent variable data.
    y (numpy array): Dependent variable data.
    linreg_flag (bool): Flag to calculate logarithmic regression statistics using linregress from scipy.
    plot_flag (bool): Flag to plot the logarithmic fit line.

    Returns:
    numpy.ndarray: Fitted logarithmic function values.
    float: Slope of the logarithmic fit line (if linreg_flag is True).
    float: Intercept of the logarithmic fit line (if linreg_flag is True).
    float: Coefficient of determination (R^2) of the logarithmic fit (if linreg_flag is True).
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
    return log_fit_fn, r_squared_log


# Power fit
def powfit(x, y, linreg_flag, plot_flag):
    """
    Fit a power-law regression model to the data and optionally calculate power-law regression statistics.

    Parameters:
    x (numpy array): Independent variable data.
    y (numpy array): Dependent variable data.
    linreg_flag (bool): Flag to calculate power-law regression statistics using linregress from scipy.
    plot_flag (bool): Flag to plot the power-law fit line.

    Returns:
    numpy.ndarray: Fitted power-law function values.
    float: Slope of the power-law fit line (if linreg_flag is True).
    float: Intercept of the power-law fit line (if linreg_flag is True).
    float: Coefficient of determination (R^2) of the power-law fit (if linreg_flag is True).
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
    return power_fit_fn, r_squared_power

# Create a DataFrame to store the fit statistics
def create_dataframe(pred_poly, r_squared_poly, linear_fit_fn, r_squared_linear, exp_fit_fn, r_squared_exp, log_fit_fn,
                     r_squared_log, power_fit_fn, r_squared_power, linreg_flag):
    # Initialize an empty DataFrame
    fits_stats = pd.DataFrame(columns=['Fit Type', 'Fitted Function Values', 'R^2', 'Slope', 'Intercept', 'Equation'])

    # Polynomial fit
    poly_dict = {'Fit Type': 'Polynomial', 'Fitted Function Values': pred_poly, 'R^2': r_squared_poly, 'Slope': None,
                 'Intercept': None, 'Equation': None}
    fits_stats = fits_stats._append(poly_dict, ignore_index=True)

    # Linear fit
    if linreg_flag:
        slope_linear, intercept_linear = linearfit(x, y, linreg_flag=True)[1:3]
        equation_linear = f'y = {slope_linear}x + {intercept_linear}'
        linear_dict = {'Fit Type': 'Linear', 'Fitted Function Values': linear_fit_fn, 'R^2': r_squared_linear,
                       'Slope': slope_linear, 'Intercept': intercept_linear, 'Equation': equation_linear}
    else:
        linear_dict = {'Fit Type': 'Linear', 'Fitted Function Values': linear_fit_fn, 'R^2': r_squared_linear,
                       'Slope': None, 'Intercept': None}
    fits_stats = fits_stats._append(linear_dict, ignore_index=True)

    # Exponential fit
    exp_dict = {'Fit Type': 'Exponential', 'Fitted Function Values': exp_fit_fn, 'R^2': r_squared_exp, 'Slope': None,
                'Intercept': None, 'Equation': None}
    fits_stats = fits_stats._append(exp_dict, ignore_index=True)

    # Logarithmic fit
    if linreg_flag:
        slope_log, intercept_log = logfit(x, y, linreg_flag=True)[1:3]
        equation_log = f'y = {slope_log}ln(x) + {intercept_log}'
        log_dict = {'Fit Type': 'Logarithmic', 'Fitted Function Values': log_fit_fn, 'R^2': r_squared_log,
                    'Slope': slope_log, 'Intercept': intercept_log, 'Equation': equation_log}
    else:
        log_dict = {'Fit Type': 'Logarithmic', 'Fitted Function Values': log_fit_fn, 'R^2': r_squared_log,
                    'Slope': None, 'Intercept': None}
    fits_stats = fits_stats._append(log_dict, ignore_index=True)

    # Power fit
    if linreg_flag:
        slope_power, intercept_power = powfit(x, y, linreg_flag=False)[1:3]
        equation_power = f'y = {intercept_power}x^{slope_power}'
        pow_dict = {'Fit Type': 'Power', 'Fitted Function Values': power_fit_fn, 'R^2': r_squared_power, 'Slope': slope_power,
                    'Intercept': intercept_power, 'Equation': equation_power}
    else:
        pow_dict = {'Fit Type': 'Power', 'Fitted Function Values': power_fit_fn, 'R^2': r_squared_power, 'Slope': None,
                    'Intercept': None}
    fits_stats = fits_stats._append(pow_dict, ignore_index=True)

    return fits_stats


def fit_portion(x, y, highest_order):
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
        fit_types = ['PolyRegres', 'linearfit', 'exponentialfit', 'logfit', 'powfit']
        fit_eqns = [f'y = {np.polyfit(x1, y1, highest_order)}',
                    f'y = {np.polyfit(x1, y1, 1)[0]}x + {np.polyfit(x1, y1, 1)[1]}',
                    f'y = {np.exp(np.polyfit(x1, np.log(y1), 1)[1])}e^{np.polyfit(x1, np.log(y1), 1)[0]}x',
                    f'y = {np.polyfit(np.log(x1), y1, 1)[0]}ln(x) + {np.polyfit(np.log(x1), y1, 1)[1]}',
                    f'y = {np.exp(np.polyfit(np.log(x1), np.log(y1), 1)[1])}x^{np.polyfit(np.log(x1), np.log(y1), 1)[0]}']
        rsqs = [Rsq(PolyRegres(x1, y1, highest_order, plot_flag=False)[0], y1),
                Rsq(linearfit(x1, y1, linreg_flag=False, plot_flag=False)[0], y1),
                Rsq(exponentialfit(x1, y1, plot_flag=False)[0], y1),
                Rsq(logfit(x1, y1, linreg_flag=False, plot_flag=False)[0], y1),
                Rsq(powfit(x1, y1, linreg_flag=False, plot_flag=False)[0], y1)]
        max_rsq1 = max(rsqs)
        best_fit_type1 = fit_types[rsqs.index(max_rsq1)]
        best_fit_eqn1 = fit_eqns[rsqs.index(max_rsq1)]

        fit_eqns = [f'y = {np.polyfit(x2, y2, highest_order)}',
                    f'y = {np.polyfit(x2, y2, 1)[0]}x + {np.polyfit(x2, y2, 1)[1]}',
                    f'y = {np.exp(np.polyfit(x2, np.log(y2), 1)[1])}e^{np.polyfit(x2, np.log(y2), 1)[0]}x',
                    f'y = {np.polyfit(np.log(x2), y2, 1)[0]}ln(x) + {np.polyfit(np.log(x2), y2, 1)[1]}',
                    f'y = {np.exp(np.polyfit(np.log(x2), np.log(y2), 1)[1])}x^{np.polyfit(np.log(x2), np.log(y2), 1)[0]}']
        rsqs = [Rsq(PolyRegres(x2, y2, highest_order, plot_flag=False)[0], y2),
                Rsq(linearfit(x2, y2, linreg_flag=False, plot_flag=False)[0], y2),
                Rsq(exponentialfit(x2, y2, plot_flag=False)[0], y2),
                Rsq(logfit(x2, y2, linreg_flag=False, plot_flag=False)[0], y2),
                Rsq(powfit(x2, y2, linreg_flag=False, plot_flag=False)[0], y2)]
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
    # Test the functions
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])

    # Polynomial regression
    pred_poly, r_squared_poly = PolyRegres(x, y, 2)

    # Linear regression
    linear_fit_fn, r_squared_linear = linearfit(x, y, linreg_flag=False)

    # Exponential regression
    exp_fit_fn, r_squared_exp = exponentialfit(x, y)

    # Logarithmic regression
    log_fit_fn, r_squared_log = logfit(x, y, linreg_flag=False)

    # Power-law regression
    pow_fit_fn, r_squared_pow = powfit(x, y, linreg_flag=False)

    fits_stats = create_dataframe(pred_poly, r_squared_poly, linear_fit_fn, r_squared_linear, exp_fit_fn, r_squared_exp,
                                  log_fit_fn, r_squared_log, pow_fit_fn, r_squared_pow, linreg_flag=False)
    print(fits_stats)
    plt.scatter(x, y, color='blue', label='Data')
    plt.legend()
    plt.show()
