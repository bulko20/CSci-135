### START OF CODE ###

# Import necessary libraries
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose as sd

# Reading the CSV file
df = pd.read_csv(r'C:\Users\HP\Dropbox\PC\Documents\College\2nd Year\2nd Sem\Synoptic Meteorology 2\Case Study\Data\CSV files for Python\Tacloban.csv')

# Convert 'YEAR', 'MONTH', and 'DAY' columns to datetime
df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])

# Set 'DATE' as the index
df.set_index('DATE', inplace=True)


# CLEANING THE DATASET -- START --


# Since -999 values indicate no observation, replace -999.0 with NaN
df['RAINFALL'].replace(-999.0, np.nan, inplace=True)
df['TMEAN'].replace(-999.0, np.nan, inplace=True)
df['TMAX'].replace(-999.0, np.nan, inplace=True)
df['TMIN'].replace(-999.0, np.nan, inplace=True)

# Replacing -1.0 with 0 since it indicates trace amounts of rainfall
df['RAINFALL'].replace(-1.0, 0, inplace=True)

# Compute overall mean and impute 'nan' values using the mean (termed as 'mean imputation')
mean_rf = df['RAINFALL'].mean()
rf_imputed = df['RAINFALL'].fillna(mean_rf)

mean_tmean = df['TMEAN'].mean()
tmean_imputed = df['TMEAN'].fillna(mean_tmean)

mean_tmax = df['TMAX'].mean()
tmax_imputed = df['TMAX'].fillna(mean_tmax)

mean_tmin = df['TMIN'].mean()
tmin_imputed = df['TMIN'].fillna(mean_tmin)

# Calculate for monthly total and average using resample to automatically identify monthly basis (uses 'M' as the argument)
totals_prcp_monthly = rf_imputed.resample('M').sum()
print(totals_prcp_monthly)

monthly_mean_tmean = tmean_imputed.resample('M').mean()
print(monthly_mean_tmean)

monthly_mean_tmax = tmax_imputed.resample('M').mean()
print(monthly_mean_tmax)

monthly_mean_tmin = tmin_imputed.resample('M').mean()
print(monthly_mean_tmin)


# CLEANING THE DATA SET -- END --



# TIME SERIES ANALYSIS FOR TEMPERATURE ANOMALIES OF ALL TEMPERATURE VARIABLES (Objective 1)


# Group the data by month using groupby and calculate the climatology 
climatology_tmean = tmean_imputed.groupby(tmean_imputed.index.month).transform('mean')
climatology_tmax = tmax_imputed.groupby(tmax_imputed.index.month).transform('mean')
climatology_tmin = tmin_imputed.groupby(tmin_imputed.index.month).transform('mean')

# Calculate the daily temperature anomalies
anomalies_tmean = tmean_imputed - climatology_tmean
anomalies_tmax = tmax_imputed - climatology_tmax
anomalies_tmin = tmin_imputed - climatology_tmin

# Calculating the monthly mean of the daily temperature anomalies using resample (uses 'M' as the argument)
monthly_mean_tmean_anomalies = anomalies_tmean.resample('M').mean()
monthly_mean_tmax_anomalies = anomalies_tmax.resample('M').mean()
monthly_mean_tmin_anomalies = anomalies_tmin.resample('M').mean()

print('\n\nMonthly Mean of Daily TMEAN Anomalies: \n', monthly_mean_tmean_anomalies,
      '\n\nMonthly Mean of Daily TMAX Anomalies: \n', monthly_mean_tmax_anomalies,
      '\n\nMonthly Mean of Daily TMIN Anomalies: \n', monthly_mean_tmin_anomalies)

# Creating a time series plot of the monthly mean of the monthly TMEAN anomalies
plt.figure(figsize=(10,6))

# Plot the anomalies
plt.plot(monthly_mean_tmean_anomalies.index, monthly_mean_tmean_anomalies, label='TMEAN')

# Fit a line to the data
zAnom1 = np.polyfit(monthly_mean_tmean_anomalies.index.to_julian_date(), monthly_mean_tmean_anomalies, 1)
pAnom1 = np.poly1d(zAnom1)

# Plot the trend line
plt.plot(monthly_mean_tmean_anomalies.index, pAnom1(monthly_mean_tmean_anomalies.index.to_julian_date()), 'r--', label='Trend')

plt.xlabel('Date')
plt.ylabel('Temperature Anomaly in °C')
plt.title('Monthly Mean of TMEAN Anomalies in Tacloban City (1991-2021)')
plt.legend()
plt.show()

# Creating a time series plot of the monthly mean of the monthly TMAX anomalies
plt.figure(figsize=(10,6))

# Plot the anomalies
plt.plot(monthly_mean_tmax_anomalies.index, monthly_mean_tmax_anomalies, label='TMAX')

# Fit a line to the data
zAnom2 = np.polyfit(monthly_mean_tmax_anomalies.index.to_julian_date(), monthly_mean_tmax_anomalies, 1)
pAnom2 = np.poly1d(zAnom2)

# Plot the trend line
plt.plot(monthly_mean_tmax_anomalies.index, pAnom2(monthly_mean_tmax_anomalies.index.to_julian_date()), 'r--', label='Trend')

plt.xlabel('Date')
plt.ylabel('Temperature Anomaly in °C')
plt.title('Monthly Mean of TMAX Anomalies in Tacloban City (1991-2021)')
plt.legend()
plt.show()

# Creating a time series plot of the monthly mean of the monthly TMIN anomalies
plt.figure(figsize=(10,6))

# Plot the anomalies
plt.plot(monthly_mean_tmin_anomalies.index, monthly_mean_tmin_anomalies, label='TMIN')

# Fit a line to the data
zAnom3 = np.polyfit(monthly_mean_tmin_anomalies.index.to_julian_date(), monthly_mean_tmin_anomalies, 1)
pAnom3 = np.poly1d(zAnom3)

# Plot the trend line
plt.plot(monthly_mean_tmin_anomalies.index, pAnom3(monthly_mean_tmin_anomalies.index.to_julian_date()), 'r--', label='Trend')

plt.xlabel('Date')
plt.ylabel('Temperature Anomaly in °C')
plt.title('Monthly Mean of TMIN Anomalies in Tacloban City (1991-2021)')
plt.legend()
plt.show()

# Seasonal decomposition residuals for temperature anomalies to use as input for Durbin-Watson test
decomp_tmean = sd(monthly_mean_tmean_anomalies, model='additive', period=12)
decomp_tmax = sd(monthly_mean_tmax_anomalies, model='additive', period=12)
decomp_tmin = sd(monthly_mean_tmin_anomalies, model='additive', period=12)


# FOR TREND ANALYSIS OF ALL METEOROLOGICAL VARIABLES USING MANN-KENDALL TEST (Objective 1)

# Import necessary libraries for MK test and autocorrelation
import pymannkendall as mk
from statsmodels.graphics.tsaplots import plot_acf as acf
from statsmodels.stats.stattools import durbin_watson as dw

# Performing Durbin-Watson test to check for autocorrelation
dw_tmean = dw(decomp_tmean.resid.dropna())
dw_tmax = dw(decomp_tmax.resid.dropna())
dw_tmin = dw(decomp_tmin.resid.dropna())

print('\n\nDurbin-Watson Test for TMEAN Anomalies: ', dw_tmean
      , '\n\nDurbin-Watson Test for TMAX Anomalies: ', dw_tmax
      , '\n\nDurbin-Watson Test for TMIN Anomalies: ', dw_tmin)

# Plotting an autocorrelation plot for the anomalies to check for autocorrelation and serve as visualization for the Durbin-Watson test
acf(monthly_mean_tmean_anomalies) # TMEAN Anomalies
plt.title('')
plt.show()

acf(monthly_mean_tmax_anomalies) # TMAX Anomalies
plt.title('')
plt.show()

acf(monthly_mean_tmin_anomalies) # TMIN Anomalies
plt.title('')
plt.show()

# Perform MK Test for Monthly Temperature Anomalies to quantitatively determine the trend
anomalies_tmean_mk = mk.seasonal_test(monthly_mean_tmean_anomalies)
print('\n\nFor tmean anomaly MK Test: ', anomalies_tmean_mk)

anomalies_tmax_mk = mk.seasonal_test(monthly_mean_tmax_anomalies)
print('\n\nFor tmax anomaly MK Test: ', anomalies_tmax_mk)

anomalies_tmin_mk = mk.seasonal_test(monthly_mean_tmin_anomalies)
print('\n\nFor tmin anomaly MK Test: ', anomalies_tmin_mk)


# FOR CORRELATION ANALYSIS AND SCATTER PLOT FOR CORRELATION VISUALIZATION (Objective 2)


# Import necessary libraries
from scipy.stats import pearsonr

# Calculate the Pearson correlation coefficient and the p-value for monthly temp vs monthly accumulated precip
corr1, p_value1 = pearsonr(monthly_mean_tmean, totals_prcp_monthly)
corr2, p_values2 = pearsonr(monthly_mean_tmax, totals_prcp_monthly)
corr3, p_values3 = pearsonr(monthly_mean_tmin, totals_prcp_monthly)

print('\n\nPearsons correlation for monthly tmean: %.3f' % corr1)
print('p-value: %.3f' % p_value1)

print('\n\nPearsons correlation for monthly tmax: %.3f' % corr2)
print('p-value: %.3f' % p_values2)

print('\n\nPearsons correlation for monthly tmin: %.3f' % corr3)
print('p-value: %.3f' % p_values3)

# Create a scatter plot of tmean vs rainfall
plt.figure()
plt.scatter(monthly_mean_tmean, totals_prcp_monthly)

# Fit a line to the data
z1 = np.polyfit(monthly_mean_tmean, totals_prcp_monthly, 1)
p1 = np.poly1d(z1)

# Plot the line of best fit
plt.plot(monthly_mean_tmean, p1(monthly_mean_tmean), 'r--')

plt.xlabel('Temperature (°C)')
plt.ylabel('Rainfall (mm)')
plt.title('Monthly Average TMEAN vs Accumulated Monthly Rainfall')
plt.show()

# Create a scatter plot of tmax vs rainfall
plt.figure()
plt.scatter(monthly_mean_tmax, totals_prcp_monthly)

# Fit a line to the data
z2 = np.polyfit(monthly_mean_tmax, totals_prcp_monthly, 1)
p2 = np.poly1d(z2)

# Plot the line of best fit
plt.plot(monthly_mean_tmax, p2(monthly_mean_tmax), 'r--')

plt.xlabel('Temperature (°C)')
plt.ylabel('Rainfall (mm)')
plt.title('Monthly Average TMAX vs Accumulated Monthly Rainfall')
plt.show()

# Create a scatter plot of tmean vs rainfall
plt.figure()
plt.scatter(monthly_mean_tmin, totals_prcp_monthly)

# Fit a line to the data
z3 = np.polyfit(monthly_mean_tmin, totals_prcp_monthly, 1)
p3 = np.poly1d(z3)

# Plot the line of best fit
plt.plot(monthly_mean_tmin, p3(monthly_mean_tmin), 'r--')

plt.xlabel('Temperature (°C)')
plt.ylabel('Rainfall (mm)')
plt.title('Monthly Average TMIN vs Accumulated Monthly Rainfall')
plt.show()

### END OF CODE ###