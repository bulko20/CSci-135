# Importing necessary libraries
import pandas as pd
from matplotlib import pyplot as plt
import pymannkendall as mk

# Importing the CSV file
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

# Compute overall mean and impute 'nan' values using the mean (called 'mean imputation')
mean_rf = df['RAINFALL'].mean()
rf_imputed = df['RAINFALL'].fillna(mean_rf)

mean_tmean = df['TMEAN'].mean()
tmean_imputed = df['TMEAN'].fillna(mean_tmean)

mean_tmax = df['TMAX'].mean()
tmax_imputed = df['TMAX'].fillna(mean_tmax)

mean_tmin = df['TMIN'].mean()
tmin_imputed = df['TMIN'].fillna(mean_tmin)


# FOR DAILY TEMPERATURE ANOMALIES (ALL TEMP VARIABLES - TMEAN, TMAX, TMIN)

# Calculate the climatology
climatology_tmean = tmean_imputed.groupby(tmean_imputed.index.month).transform('mean')
climatology_tmax = tmax_imputed.groupby(tmax_imputed.index.month).transform('mean')
climatology_tmin = tmin_imputed.groupby(tmin_imputed.index.month).transform('mean')

# Calculate the daily temperature anomalies
anomalies_tmean = tmean_imputed - climatology_tmean
anomalies_tmax = tmax_imputed - climatology_tmax
anomalies_tmin = tmin_imputed - climatology_tmin

print('TMEAN:', anomalies_tmean, 'TMAX: ', anomalies_tmax, 'TMIN: ', anomalies_tmin)

# Creating a color map for all temperature values
colors_tmean = ['blue' if value < 0 else 'red' for value in anomalies_tmean]
colors_tmax = ['blue' if value < 0 else 'red' for value in anomalies_tmax]
colors_tmin = ['blue' if value < 0 else 'red' for value in anomalies_tmin]

# Plot the daily temp anomalies with the color map
plt.figure(figsize=(10,6))
plt.bar(anomalies_tmean.index, anomalies_tmean, color=colors_tmean)
plt.xlabel('Date')
plt.ylabel('Temperature Anomaly in °C')
plt.title('Daily Average TMEAN Anomalies in Tacloban City (1991-2021)')
plt.show()

plt.figure(figsize=(10,6))
plt.bar(anomalies_tmax.index, anomalies_tmax, color=colors_tmax)
plt.xlabel('Date')
plt.ylabel('Temperature Anomaly in °C')
plt.title('Daily Average TMAX Anomalies in Tacloban City (1991-2021)')
plt.show()

plt.figure(figsize=(10,6))
plt.bar(anomalies_tmin.index, anomalies_tmin, color=colors_tmin)
plt.xlabel('Date')
plt.ylabel('Temperature Anomaly in °C')
plt.title('Daily Average TMIN Anomalies in Tacloban City (1991-2021)')
plt.show()

# Perform MK Test for Daily Temperature Anomalies
anomalies_tmean_mk = mk.original_test(anomalies_tmean)
print('\n\nFor tmean anomaly MK Test: ', anomalies_tmean_mk)

anomalies_tmax_mk = mk.original_test(anomalies_tmax)
print('\n\nFor tmax anomaly MK Test: ', anomalies_tmax_mk)

anomalies_tmin_mk = mk.original_test(anomalies_tmin)
print('\n\nFor tmin anomaly MK Test: ', anomalies_tmin_mk)

### END OF CODE ###