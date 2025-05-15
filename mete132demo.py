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

# Since -999 values indicate no observation, replace -999.0 with NaN
df['RAINFALL'].replace(-999.0, np.nan, inplace=True)
df['TMEAN'].replace(-999.0, np.nan, inplace=True)
df['TMAX'].replace(-999.0, np.nan, inplace=True)
df['TMIN'].replace(-999.0, np.nan, inplace=True)

# Replacing -1.0 with 0 since it indicates trace amounts of rainfall
df['RAINFALL'].replace(-1.0, 0, inplace=True)

# Compute overall mean and impute 'nan' values using the mean (mean imputation)
mean_rf = df['RAINFALL'].mean()
rf_imputed = df['RAINFALL'].fillna(mean_rf)

mean_tmean = df['TMEAN'].mean()
tmean_imputed = df['TMEAN'].fillna(mean_tmean)

mean_tmax = df['TMAX'].mean()
tmax_imputed = df['TMAX'].fillna(mean_tmax)

mean_tmin = df['TMIN'].mean()
tmin_imputed = df['TMIN'].fillna(mean_tmin)

# Plot the autocorrelation
acf(rf_imputed)
plt.title('Autocorrelation for RF')
plt.show()

acf(tmean_imputed)
plt.title('Autocorrelation for TMEAN')
plt.show()

acf(tmax_imputed)
plt.title('Autocorrelation for TMAX')
plt.show()

acf(tmin_imputed)
plt.title('Autocorrelation for TMIN')
plt.show()

# Conduct seasonal decomposition
rf_sd = sd(rf_imputed, model='additive', period=12)
rf_sd.plot()

tmean_sd = sd(tmean_imputed, model='additive', period=12)
tmean_sd.plot()

tmax_sd = sd(tmax_imputed, model='additive', period=12)
tmax_sd.plot()

tmin_sd = sd(tmin_imputed, model='additive', period=12)
tmin_sd.plot()

# Calculate for monthly total and average using resample to automatically identify monthly (M) basis
totals_prcp_monthly = rf_imputed.resample('M').sum()
print(totals_prcp_monthly)

monthly_mean_tmean = tmean_imputed.resample('M').mean()
print(monthly_mean_tmean)

monthly_mean_tmax = tmax_imputed.resample('M').mean()
print(monthly_mean_tmax)

monthly_mean_tmin = tmin_imputed.resample('M').mean()
print(monthly_mean_tmin)

# Calculate for annual total and average using resample to automatically identify yearly (Y) basis
annual_prcp = rf_imputed.resample('Y').sum()
print(annual_prcp)

annual_tmean = tmean_imputed.resample('Y').mean()
print(annual_tmean)

annual_tmax = tmax_imputed.resample('Y').mean()
print(annual_tmax)

annual_tmin = tmin_imputed.resample('Y').mean()
print(annual_tmin)

# Plotting annual total for precipitation
plt.figure()
plt.plot(annual_prcp.index, annual_prcp.values, **{'color': 'blue', 'marker': 'o'})
plt.title('Annual Total Precipitation (1991-2021)')
plt.xlabel('Years')
plt.ylabel('Precipitaiton (in mm)')
plt.show()

# Import necessary libraries for MK test
import pymannkendall as mk

# Perform Mann-Kendall test for monthly averages
result_tmean = mk.original_test(monthly_mean_tmean)
result_tmax = mk.original_test(monthly_mean_tmax)
result_tmin = mk.original_test(monthly_mean_tmin)
result_prcp = mk.original_test(totals_prcp_monthly)

print('For TMEAN: \n', result_tmean, '\n\nFor rainfall: \n', result_prcp, '\n\nFor TMAX: \n', result_tmax, '\n\nFor TMIN: \n', result_tmin)

# Import necessary libraries
from scipy.stats import pearsonr

# Calculate the Pearson correlation coefficient and the p-value for monthly temp vs monthly accumulated precip
corr1, p_value1 = pearsonr(monthly_mean_tmean, totals_prcp_monthly)

print('\n\nPearsons correlation for monthly averages: %.3f' % corr1)
print('p-value: %.3f' % p_value1)

# Create a scatter plot of temperature vs rainfall
plt.figure()
plt.scatter(monthly_mean_tmean, totals_prcp_monthly)

# Fit a line to the data
z = np.polyfit(monthly_mean_tmean, totals_prcp_monthly, 1)
p = np.poly1d(z)

# Plot the line of best fit
plt.plot(monthly_mean_tmean, p(monthly_mean_tmean), 'r--')

plt.xlabel('Temperature (°C)')
plt.ylabel('Rainfall (mm)')
plt.title('Monthly Average Temperature vs Accumulated Monthly Rainfall')
plt.show()

# Calculate the climatology
climatology_tmean = tmean_imputed.groupby(tmean_imputed.index.month).transform('mean')
climatology_tmax = tmax_imputed.groupby(tmax_imputed.index.month).transform('mean')
climatology_tmin = tmin_imputed.groupby(tmin_imputed.index.month).transform('mean')


# Calculate the daily temperature anomalies
anomalies_tmean = tmean_imputed - climatology_tmean
anomalies_tmax = tmax_imputed - climatology_tmax
anomalies_tmin = tmin_imputed - climatology_tmin

print('TMEAN:', anomalies_tmean, 'TMAX: ', anomalies_tmax, 'TMIN: ', anomalies_tmin)


### RECODE MONTHLY TEMPERATURE ANOMALIES

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

# Creating a color map for all temperature values
colors_tmean = ['blue' if value < 0 else 'red' for value in anomalies_tmean]
colors_tmax = ['blue' if value < 0 else 'red' for value in anomalies_tmax]
colors_tmin = ['blue' if value < 0 else 'red' for value in anomalies_tmin]

# Plot the daily tmean anomalies with the color map
plt.figure(figsize=(10,6))
plt.bar(anomalies_tmean.index, anomalies_tmean, color=colors_tmean)
plt.xlabel('Date')
plt.ylabel('Temperature Anomaly in °C')
plt.title('Daily Average Temperature Anomalies in Tacloban City (1991-2021)')
plt.show()

# Perform MK Test for Daily Temperature Anomalies
anomalies_tmean_mk = mk.original_test(anomalies_tmean)
print('\n\nFor tmean anomaly MK Test: ', anomalies_tmean_mk)

# Perform MK Test for Monthly Average of Temperature Anomalies
# tAnom_mk = mk.original_test(tAnom_monthly)
# print('\n\nFor temperature anomaly MK Test: ', tAnom_mk)


# Resample to monthly frequency, if necessary
tmean_monthly = tmean_imputed.resample('M').mean()
tmax_monthly = tmax_imputed.resample('M').mean()
tmin_monthly = tmin_imputed.resample('M').mean()

# Calculate the climatological mean per month
climatology_tmean = tmean_monthly.groupby(tmean_monthly.index.month).mean()
climatology_tmax = tmax_monthly.groupby(tmax_monthly.index.month).mean()
climatology_tmin = tmin_monthly.groupby(tmin_monthly.index.month).mean()


# Initialize empty DataFrames to store the anomalies
anomalies_tmean = pd.DataFrame(index=tmean_monthly.index)
anomalies_tmax = pd.DataFrame(index=tmax_monthly.index)
anomalies_tmin = pd.DataFrame(index=tmin_monthly.index)

# Calculate the anomalies for each month
for month in range(1, 13):
    anomalies_tmean.loc[tmean_monthly.index.month == month, 'value'] = (tmean_monthly[tmean_monthly.index.month == month] - climatology_tmean[month]).values
    anomalies_tmax.loc[tmax_monthly.index.month == month, 'value'] = (tmax_monthly[tmax_monthly.index.month == month] - climatology_tmax[month]).values
    anomalies_tmin.loc[tmin_monthly.index.month == month, 'value'] = (tmin_monthly[tmin_monthly.index.month == month] - climatology_tmin[month]).values

# Plot the anomalies
plt.plot(anomalies_tmean.index, anomalies_tmean['value'])
plt.xlabel('Date')
plt.ylabel('Temperature Anomaly in °C')
plt.title('Monthly TMEAN Anomalies in Tacloban City (1991-2021)')
plt.show()

# Plot the anomalies
plt.plot(anomalies_tmax.index, anomalies_tmax['value'])
plt.xlabel('Date')
plt.ylabel('Temperature Anomaly in °C')
plt.title('Monthly TMAX Anomalies in Tacloban City (1991-2021)')
plt.show()

# Plot the anomalies
plt.plot(anomalies_tmin.index, anomalies_tmin['value'])
plt.xlabel('Date')
plt.ylabel('Temperature Anomaly in °C')
plt.title('Monthly TMIN Anomalies in Tacloban City (1991-2021)')
plt.show()