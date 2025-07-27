import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

# Step 1: Load the .mat files
mat_file1 = 'matlab/outs.mat' 
mat_file2 = 'matlab/SP500.mat'
data1 = sio.loadmat(mat_file1)
data2 = sio.loadmat(mat_file2)

# Assume the time series are stored in variables 'series1' and 'series2'
series1 = data1['outs'].flatten()  # S&P 500 values
series2 = data2['SP500'].flatten()  # Early warning signals

# Step 2: Prepare dates
start_date = datetime(2012, 3, 28)
n_points = len(series1)  # Assuming both series have the same length
date_range = [start_date + timedelta(days=i) for i in range(n_points)]
dates = pd.to_datetime(date_range)

# Step 3: Detect early warnings
threshold = np.percentile(series2, 99)  # 99th percentile as threshold
early_warning_dates = dates[series2 >= threshold]

# Step 4: Plot the graph
plt.figure(figsize=(12, 6))
plt.plot(dates, series1, label='S&P 500', color='blue', linewidth=1.2)

# Add vertical lines for early warnings
for date in early_warning_dates:
    plt.axvline(x=date, color='orange', linestyle='--', alpha=0.7)

plt.title('S&P500 and detected early warnings', fontsize=16, fontweight='bold')
plt.ylabel('Price', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Save or display the plot
plt.savefig('s&p500_with_early_warnings.png')
plt.show()