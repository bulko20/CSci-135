# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 18:45:09 2024

@author: HP
"""

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots


# Load your data
df = pd.read_csv(r'C:\Users\HP\Dropbox\PC\Documents\College\2nd Year\2nd Sem\Synoptic Meteorology 2\Case Study\Data\CSV files for Python\Tacloban.csv')

# Load the data from your CSV into a list focused only on RAINFALL column
x = df['RAINFALL']

# Calculate autocorrelation
acf = sm.tsa.acf(x)

# Print the autocorrelation
print(acf)

# Plot the autocorrelation
fig = tsaplots.plot_acf(x, lags=30)
plt.show()