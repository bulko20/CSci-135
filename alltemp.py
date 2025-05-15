# Import necessary libraries
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt

# Reading the CSV file
df = pd.read_csv(r'C:\Users\HP\Dropbox\PC\Documents\College\2nd Year\2nd Sem\Synoptic Meteorology 2\Case Study\Data\CSV files for Python\Tacloban.csv')

# Convert 'YEAR', 'MONTH', and 'DAY' columns to datetime
df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])

# Replace -999 with NaN
df['TMEAN'].replace(-999, np.nan, inplace=True)
df['TMAX'].replace(-999, np.nan, inplace=True)
df['TMIN'].replace(-999, np.nan, inplace=True)

# Set 'DATE' as the index
df.set_index('DATE', inplace=True)

# Interpolate missing values
df['TMEAN'] = df['TMEAN'].interpolate(method='time')
df['TMAX'] = df['TMAX'].interpolate(method='time')
df['TMIN'] = df['TMIN'].interpolate(method='time')

# Calculate monthly averages
monthly_ave_tmean = df.resample('M').mean()['TMEAN']
monthly_ave_tmax = df.resample('M').mean()['TMAX']
monthly_ave_tmin = df.resample('M').mean()['TMIN']

print('TMEAN: ', monthly_ave_tmean)
print('TMAX: ', monthly_ave_tmax)
print('TMIN: ', monthly_ave_tmin)

# To plot the monthly averages of the temperature
plt.figure()
plt.title('Monthly Averages of the TMEAN, TMAX, TMIN')
plt.xlabel('Years')
plt.ylabel('Temperature (°C)')
plt.plot(monthly_ave_tmean.index, monthly_ave_tmean.values, color='green', label='TMEAN')
plt.plot(monthly_ave_tmax.index, monthly_ave_tmax.values, color='red', label='TMAX')
plt.plot(monthly_ave_tmin.index, monthly_ave_tmin.values, color='blue', label='TMIN')
plt.show()

# For seasonal decompose
from statsmodels.tsa.seasonal import seasonal_decompose as sd

tmean_sd = sd(monthly_ave_tmean, model='additive', period=12)
tmean_sd.plot()
plt.show()

tmax_sd = sd(monthly_ave_tmax, model='additive', period=12)
tmax_sd.plot()
plt.show()

tmin_sd = sd(monthly_ave_tmin, model='additive', period=12)
tmin_sd.plot()
plt.show()


# Calculate the climatology
clima_tmean = df.groupby(df.index.month)['TMEAN'].transform('mean')
clima_tmax = df.groupby(df.index.month)['TMAX'].transform('mean')
clima_tmin = df.groupby(df.index.month)['TMIN'].transform('mean')

print('TMEAN Normal: ', clima_tmean)
print('TMAX Normal: ', clima_tmax)
print('TMIN Normal: ', clima_tmin)

# Calculating the anomalies
anomalies_tmean = df['TMEAN'] - clima_tmean
anomalies_tmax = df['TMAX'] - clima_tmax
anomalies_tmin = df['TMIN'] - clima_tmin

print('TMEAN Anoamlies: ', anomalies_tmean)
print('TMAX Anoamlies: ', anomalies_tmax)
print('TMIN Anoamlies: ', anomalies_tmin)

# Monthly anomalies of temperature
tmeanAnom_filter = anomalies_tmean['1991':'2021']
tmeanAnom_monthly = tmeanAnom_filter.resample('M').mean()

tmaxAnom_filter = anomalies_tmax['1991':'2021']
tmaxAnom_monthly = tmaxAnom_filter.resample('M').mean()

tminAnom_filter = anomalies_tmin['1991':'2021']
tminAnom_monthly = tminAnom_filter.resample('M').mean()

print(tmeanAnom_monthly)
print(tmaxAnom_monthly)
print(tminAnom_monthly)

# To plot the monthly averages of the temperature anomalies
plt.figure()
plt.plot(tmeanAnom_monthly.index, tmeanAnom_monthly.values, color='green')
plt.plot(tmaxAnom_monthly.index, tmaxAnom_monthly.values, color='red')
plt.plot(tminAnom_monthly.index, tminAnom_monthly.values, color='blue')
plt.show()
