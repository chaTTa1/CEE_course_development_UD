"""
Author: Alex Chattos (chattosa1@udayton.edu)

University of Dayton Department of Civil and Environmental Engineering


This script uses fit_lines_and_statistics module to predict traffic flow models with observed data

Required packages/modules:
    -numpy
    -matplotlib
    -pandas
    -fit_lines_and_statistics (our created module for fit lines, it should be in the same directory as this script)
    numpy, matplotlib, and pandas should be installed with Spyder IDE installation

"""


import numpy as np
import pandas as pd
import fit_lines_and_statistics

"""
- use pandas to read data from excel
- make sure excel file is in the same directory or give complete file path for 'read_excel' function
- when creating numpy arrays for densities and speeds, be sure that the column headers are EXACTLY the same as the headers in the excel files
"""
df = pd.read_excel('Densities_Speed.xlsx')

# array for densities, D (veh, mi/h)
densities = np.array(df['densities'].values) # make sure header name matches the one in the Excel file

# array for speed, S (mi/h)
speeds = np.array(df['speeds'].values) # make sure header name matches the one in the Excel file

"""
Use our created module to calculate and plot fit lines



inputs:
    - densities as independent variable
    - speeds as dependent variable
    - linreg_flag(bool): (if enabled, will return linear regression stats using scipy package [see outputs])
outputs:
    - fitted function values
    - coefficients of determination (R^2) (automatically for exponential fit, optional for other fit lines with linreg_flag)
    - slope of the fit line (if linreg_flag is True).
    - intercept of the power-law fit line (if linreg_flag is True).
"""


# plot polynomial fit
pred_poly, r_squared_poly, equation_poly = fit_lines_and_statistics.PolyRegress(densities, speeds, 2, plot_flag=True)

# plot linear fit
linear_fit_fn, r_squared_linear, equation_linear = fit_lines_and_statistics.linearfit(densities, speeds, plot_flag=True)

# plot exponential fit
exp_fit_fn, r_squared_exp, equation_exp = fit_lines_and_statistics.exponentialfit(densities, speeds, plot_flag=True)

# plot logarithmic fit
log_fit_fn, r_squared_log, equation_log = fit_lines_and_statistics.logfit(densities, speeds, plot_flag=True)

# plot power fit
power_fit_fn, r_squared_power, equation_power = fit_lines_and_statistics.powfit(densities, speeds, plot_flag=True)

# create data frame for fit lines statistics
fits_stats = fit_lines_and_statistics.create_dataframe(pred_poly,  linear_fit_fn,  exp_fit_fn, log_fit_fn, power_fit_fn,
                                                     r_squared_poly, r_squared_linear, r_squared_exp, r_squared_log, r_squared_power,
                                                     equation_linear, equation_log, equation_power, equation_exp, equation_poly)

pd.set_option('display.max_columns', None)

fit_lines_and_statistics.fit_portion(densities, speeds, 2)
# Print the DataFrame
print(fits_stats)

