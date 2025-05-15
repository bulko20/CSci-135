import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
import pymannkendall as mk
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson as dw
from statsmodels.tsa.stattools import adfuller

# Reading the CSV file
x = pd.read_csv(r'C:\Users\CET_DMET-07\Desktop\CSci 135 - Bulic\data.csv')

# Creating a data frame
df = pd.DataFrame(x[['DATE', 'PRCP', 'TEMP']])

# Converting to series
rf = pd.Series(df['PRCP'])
temp = pd.Series(df['TEMP'])
date = pd.Series(df['DATE'])


# FOR RAINFALL
# Creating a plot
plt.plot(rf.index, rf.values)
plt.tick_params(axis='x', rotation = 90)
plt.xlabel('Dates')
plt.ylabel('Rainfall (in mm)')
plt.title('Time Series of Rainfall (2023)')

# Grouping the data values per 5 days to create labels in x-axis
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))  
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d')) 

plt.show()

# To check for autocorrelation
ac_rf = acf(rf)
plot_acf(rf, lags=100)
plt.title('RAINFALL ACF')
plt.show()

# For seasonal decompose
rf_decompose = seasonal_decompose(rf, model='additive', period=30)
rf_decompose.plot()
plt.xticks()
plt.show()

# FOR TEMPERATURE
# Creating a plot
plt.plot(temp.index, temp.values)
plt.tick_params(axis='x', rotation = 90)
plt.xlabel('Dates')
plt.ylabel('Temperature (in °C)')
plt.title('Time Series of Temperature (2023)')

# Grouping the data values per 5 days to create labels in x-axis
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))  
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d')) 

plt.show()

# To check for autocorrelation
ac_rf = acf(temp)
plot_acf(temp, lags=20)
plt.title('TEMPERATURE ACF')
plt.show()

# For seasonal decompose
temp_decompose = seasonal_decompose(temp, model='additive', period=30)
temp_decompose.plot()
plt.xticks()
plt.show()

# To conduct MK test
mk_prcp = mk.original_test(rf)
print('MK for PRCP\n', mk_prcp)

mk_temp = mk.original_test(temp)
print('\nMK for TEMP\n ', mk_temp)

# Durbin Watson Test
Y_temp = df['TEMP']
X_temp = list(range(0, len(df['TEMP']), 1))
temp_model = sm.OLS(Y_temp, X_temp).fit()
temp_dw = dw(temp_model.resid)

Y_prcp = df['PRCP']
X_prcp = list(range(0, len(df['PRCP']), 1))
prcp_model = sm.OLS(Y_prcp, X_prcp).fit()
prcp_dw = dw(prcp_model.resid)

# Augmented Dickey-Fuller Test
adftest = adfuller(df['TEMP'], autolag='AIC', regression='ct')
print("ADF Test Results")
print("Null Hypothesis: The series has a unit root (non-stationary)")
print("ADF-Statistic:", adftest[0])
print("P-Value:", adftest[1])
print("Number of lags:", adftest[2])
print("Number of observations:", adftest[3])
print("Critical Values:", adftest[4])
print("Note: If P-Value is smaller than 0.05, we reject the null hypothesis and the series is stationary")