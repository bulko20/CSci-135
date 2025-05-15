### START OF CODE ###

# Import necessary libraries
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose as sd
from statsmodels.graphics.tsaplots import plot_acf as acf

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

# CLEANING THE DATA SET -- END --


# TIME SERIES ANALYSIS FOR ALL TEMP VARIABLES AND RAINFALL (Objective 1.a)


# Calculate for monthly total and average using resample to automatically identify monthly basis (uses 'M' as the argument)
totals_prcp_monthly = rf_imputed.resample('M').sum()
print(totals_prcp_monthly)

monthly_mean_tmean = tmean_imputed.resample('M').mean()
print(monthly_mean_tmean)

monthly_mean_tmax = tmax_imputed.resample('M').mean()
print(monthly_mean_tmax)

monthly_mean_tmin = tmin_imputed.resample('M').mean()
print(monthly_mean_tmin)

# Conducting seasonal decomposition to use the residuals for Durbin-Watson test
dp_rf = sd(totals_prcp_monthly, model='additive', period=12)
dp_tmean = sd(monthly_mean_tmean, model='additive', period=12)
dp_tmax = sd(monthly_mean_tmax, model='additive', period=12)
dp_tmin = sd(monthly_mean_tmin, model='additive', period=12)

# Using Durbin-Watson test to check for autocorrelation
from statsmodels.stats.stattools import durbin_watson as dw

# Calculate the Durbin-Watson statistic for the monthly total rainfall
dw_rf = dw(dp_rf.resid.dropna())
dw_tmean = dw(dp_tmean.resid.dropna())
dw_tmax = dw(dp_tmax.resid.dropna())
dw_tmin = dw(dp_tmin.resid.dropna())

print('\n\nDurbin-Watson Test for RF:', dw_rf,
      '\n\nDurbin-Watson Test for TMEAN:', dw_tmean,
      '\n\nDurbin-Watson Test for TMAX:', dw_tmax,
      '\n\nDurbin-Watson Test for TMIN:', dw_tmin)  ## NOTE: Values printed will be between 0-4... 2 = no autocorr, < 2 = negative autocorr, > 2 = positive autocorr

# Plot the autocorrelation (lags=36 to provide a bigger picture of the data)
acf(totals_prcp_monthly, lags=36)
plt.title('Autocorrelation for RF')
plt.show()

acf(monthly_mean_tmean, lags=36)
plt.title('Autocorrelation for TMEAN')
plt.show()

acf(monthly_mean_tmax, lags=36)
plt.title('Autocorrelation for TMAX')
plt.show()

acf(monthly_mean_tmin, lags=36)
plt.title('Autocorrelation for TMIN')
plt.show()


# TIME SERIES ANALYSIS FOR TEMPERATURE ANOMALIES OF ALL TEMPERATURE VARIABLES (Objective 1.b)


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
plt.title('Monthly Mean of Daily TMEAN Anomalies in Tacloban City (1991-2021)')
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
plt.title('Monthly Mean of Daily TMAX Anomalies in Tacloban City (1991-2021)')
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
plt.title('Monthly Mean of Daily TMIN Anomalies in Tacloban City (1991-2021)')
plt.legend()
plt.show()

# Seasonal decomposition plots for temperature anomalies
decomp_tmean = sd(monthly_mean_tmean_anomalies, model='additive', period=12)
decomp_tmax = sd(monthly_mean_tmax_anomalies, model='additive', period=12)
decomp_tmin = sd(monthly_mean_tmin_anomalies, model='additive', period=12)

fig1 = decomp_tmean.plot()
fig1.axes[0].set_title('TMEAN Anomalies')
plt.show()

fig2 = decomp_tmax.plot()
fig2.axes[0].set_title('TMAX Anomalies')
plt.show()

fig3 = decomp_tmin.plot()
fig3.axes[0].set_title('TMIN Anomalies')
plt.show()


# FOR TREND ANALYSIS OF ALL METEOROLOGICAL VARIABLES USING MANN-KENDALL TEST (Objective 2)


# Import necessary libraries for MK test
import pymannkendall as mk

# Perform MK Test for Monthly Temperature Anomalies to quantitatively determine the trend
anomalies_tmean_mk = mk.original_test(monthly_mean_tmean_anomalies)
print('\n\nFor tmean anomaly MK Test: ', anomalies_tmean_mk)

anomalies_tmax_mk = mk.original_test(monthly_mean_tmax_anomalies)
print('\n\nFor tmax anomaly MK Test: ', anomalies_tmax_mk)

anomalies_tmin_mk = mk.original_test(monthly_mean_tmin_anomalies)
print('\n\nFor tmin anomaly MK Test: ', anomalies_tmin_mk)

# Perform Mann-Kendall test for monthly averages to quantitatively determine the trend
result_tmean = mk.seasonal_test(monthly_mean_tmean)
result_tmax = mk.seasonal_test(monthly_mean_tmax)
result_tmin = mk.seasonal_test(monthly_mean_tmin)
result_prcp = mk.original_test(totals_prcp_monthly)

print('\n\nFor rainfall: \n', result_prcp, 
      '\n\nFor TMEAN: \n', result_tmean, 
      '\n\nFor TMAX: \n', result_tmax, 
      '\n\nFor TMIN: \n', result_tmin)


# FOR CORRELATION ANALYSIS AND SCATTER PLOT FOR CORRELATION VISUALIZATION (Objective 3)


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
