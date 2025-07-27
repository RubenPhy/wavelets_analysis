import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns
from matplotlib.lines import Line2D
import os

# Configuración global de Seaborn y Matplotlib
sns.set_theme(style="darkgrid")
plt.rcParams.update({"axes.titleweight": "bold"})

# Step 1: Load the .mat files
mat_file1 = 'matlab/outs.mat' 
mat_file2 = 'matlab/SP500.mat'
data1 = sio.loadmat(mat_file1)
data2 = sio.loadmat(mat_file2)

# Assume the time series are stored in variables 'series1' and 'series2'
series1 = data1['outs'].flatten()  # Early warning signals
series2 = data2['SP500'].flatten()  # S&P 500 values

# Step 2: Prepare dates
# Load dates from S&P 500 Historical Data.csv
csv_file = 'S&P 500 Historical Data.csv'
df = pd.read_csv(csv_file, parse_dates=['Date'])
dates = df['Date'].iloc[:len(series1)].reset_index(drop=True)
dates = dates[::-1]
# Suma 100 dias a las fechas para alinear con los datos de 'outs'
dates = [date + timedelta(days=-450) for date in dates]

# Step 4: Plot the graph with two y-axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot S&P 500 on the left y-axis
series2_norm = series2 / np.max(series2)
ax1.plot(dates, series2_norm, label='S&P 500 (normalized)', color='blue', linewidth=1.2)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('S&P 500 (normalized)', fontsize=12, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a second y-axis for the 'outs' series
ax2 = ax1.twinx()
ax2.plot(dates, series1, label='Early Warnings', color='orange', linewidth=1.0, alpha=0.7)
ax2.set_ylabel('Early Warning Signal', fontsize=12, color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Customize the plot
plt.title('S&P500 and detected early warnings', fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()

# Save or display the plot
plt.savefig('plots/SP500-outliers_python.png')
plt.show()

##################################

# Step 1: Load the .mat files
mat_file1 = 'matlab/variabsig4.mat' 
data1 = sio.loadmat(mat_file1)

# Assume the time series are stored in variables 'series1' and 'series2'
series1 = data1['variabsig4'].flatten()  # Early warning signals

# Step 4: Plot the graph with two y-axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot S&P 500 
ax1.plot(dates, series1, label='S&P 500 (normalized)', color='blue', linewidth=1.2)

# Customize the plot
plt.title('Normalized variability of the S&P500', fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()

# Save or display the plot
plt.savefig('plots/nvar-SP500_python.png')
plt.show()