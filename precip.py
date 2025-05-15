# Import necessary libraries
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt

# Reading the CSV file
df = pd.read_csv(r'C:\Users\HP\Dropbox\PC\Documents\College\2nd Year\2nd Sem\Synoptic Meteorology 2\Case Study\Data\CSV files for Python\Tacloban.csv')

# Convert 'YEAR', 'MONTH', and 'DAY' columns to datetime
df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])

# Set 'DATE' as the index
df.set_index('DATE', inplace=True)

# Replace -999 with NaN
df['RAINFALL'].replace(-999.0, np.nan, inplace=True)
df['TMEAN'].replace(-999.0, np.nan, inplace=True)
df['TMAX'].replace(-999.0, np.nan, inplace=True)
df['TMIN'].replace(-999.0, np.nan, inplace=True)
df['RAINFALL'].replace(-1.0, 0, inplace=True)

# Interpolate missing values of temperature using linear interpolation
df['TMEAN'] = df['TMEAN'].interpolate(method='linear')
df['TMAX'] = df['TMAX'].interpolate(method='linear')
df['TMIN'] = df['TMIN'].interpolate(method='linear')

# Interpolate missing values of rainfall using mean imputation
mean_rf = df['RAINFALL'].mean()
rf_imputed = df['RAINFALL'].fillna(mean_rf)

# Calculate monthly averages
monthly_averages_temp = df.resample('M').mean()['TMEAN']

print(monthly_averages_temp)

# Plot the time series data
plt.plot(monthly_averages_temp.index, monthly_averages_temp)
plt.xlabel('Date')
plt.ylabel('Temperature in °C')
plt.title('Monthly Average Temperature (1991-2021)')
plt.legend()
plt.show()

# Calculate accumulated precipitation monthly
monthly_totals_prcp = rf_imputed.resample('M').sum()
print(monthly_totals_prcp)

# Plot the time series data
plt.plot(monthly_totals_prcp.index, monthly_totals_prcp)
plt.xlabel('Date')
plt.ylabel('Rainfall in mm')
plt.title('Accumulated Monthly Rainfall (1991-2021)')
plt.legend()
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf as acf

# FOR TEMPERATURE
# Check for autocorrelation
acf(monthly_averages_temp, lags=12)
plt.show()

# Check for seasonality
result = seasonal_decompose(monthly_averages_temp, model='additive', period=12)
result.plot()
plt.show()

# FOR RAINFALL
# Check for autocorrelation
acf(monthly_totals_prcp, lags=12)
plt.show()

#Check for seasonality
result = seasonal_decompose(monthly_totals_prcp, model='additive', period=12)
result.plot()
plt.show()


# Import necessary libraries
import pymannkendall as mk

# Perform Mann-Kendall test for monthly averages
result_temp = mk.original_test(monthly_averages_temp)
result_prcp = mk.original_test(monthly_totals_prcp)

print('For temperature: \n', result_temp, '\n\n','For rainfall: \n', result_prcp)

# Import necessary libraries
from scipy.stats import pearsonr

# Calculate the Pearson correlation coefficient and the p-value for monthly temp vs monthly accumulated precip
corr1, p_value1 = pearsonr(monthly_averages_temp, monthly_totals_prcp)

print('\n\nPearsons correlation for monthly averages: %.3f' % corr1)
print('p-value: %.3f' % p_value1)

# Create a scatter plot of temperature vs rainfall
plt.figure(figsize=(10,6))
plt.scatter(monthly_averages_temp, monthly_totals_prcp)

# Fit a line to the data
z = np.polyfit(monthly_averages_temp, monthly_totals_prcp, 1)
p = np.poly1d(z)

# Plot the line of best fit
plt.plot(monthly_averages_temp, p(monthly_averages_temp), 'r--')

plt.xlabel('Temperature (°C)')
plt.ylabel('Rainfall (mm)')
plt.title('Monthly Average Temperature vs Accumulated Monthly Rainfall')
plt.show()

# Calculate the climatology
climatology_temp = df.groupby(df.index.month)['TMEAN'].transform('mean')
climatology_prcp = df.groupby(df.index.month)['RAINFALL'].transform('mean')

# Calculate the daily temperature anomalies
anomalies_temp = df['TMEAN'] - climatology_temp
#anomalies_prcp = df['RAINFALL'] - climatology_prcp

print(anomalies_temp)

# Create a color map where values below 0 are blue and above 0 are red
colors = ['blue' if value < 0 else 'red' for value in anomalies_temp]

# Plot the daily temperature anomalies with the color map
plt.figure(figsize=(10,6))
plt.bar(anomalies_temp.index, anomalies_temp, color=colors)
plt.xlabel('Date')
plt.ylabel('Temperature Anomaly in °C')
plt.title('Daily Average Temperature Anomalies in Tacloban City (1991-2021)')
plt.show()

# Perform MK Test for Daily Temperature Anomalies
anomalies_mk = mk.original_test(anomalies_temp)
print('\n\nFor temperature anomaly MK Test: ', anomalies_mk)

### RECODE MONTHLY TEMPERATURE ANOMALIES -- START --

# Calculate the average of the monthly values from 1991 to 2021 of the temperature anomalies
# tAnom_filter = anomalies_temp['1991':'2021']
# tAnom_monthly = tAnom_filter.resample('M').mean()

# Create a color map where values below 0 are blue and above 0 are red
# colors_tAnom = ['blue' if value < 0 else 'red' for value in tAnom_monthly]

# Plot the monthly average of temperature anomalies with the color map
# plt.figure()
# plt.bar(tAnom_monthly.index, tAnom_monthly.values, color=colors_tAnom)
# plt.xlabel('Date')
# plt.ylabel('Temperature Anomaly in °C')
# plt.title('Monthly Average of Temperature Anomalies in Tacloban City (1991-2021)')
# plt.show()

# print(tAnom_monthly.index)
# print(tAnom_monthly.values)

### RECODE MONTHLY TEMPERATURE ANOMALIES -- END --

# Perform MK Test for Monthly Average of Temperature Anomalies
# tAnom_mk = mk.original_test(tAnom_monthly)
# print('\n\nFor temperature anomaly MK Test: ', tAnom_mk)