# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 20:57:49 2024

@author: 91979
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic data for each variable
num_samples = 56000
data = {
    'Fault': np.random.choice([0, 1, 2, 3], num_samples),
    'MAP (kPa)': np.random.uniform(60, 120, num_samples),
    'TPS (%)': np.random.uniform(0, 100, num_samples),
    'Force (N)': np.random.uniform(100, 500, num_samples),
    'Power (kW)': np.random.uniform(50, 200, num_samples),
    'RPM': np.random.uniform(1000, 7000, num_samples),
    'Fuel consumption l/h': np.random.uniform(5, 20, num_samples),
    'Fuel consumption l/100km': np.random.uniform(5, 15, num_samples),
    'Speed (km/h)': np.random.uniform(0, 200, num_samples),
    'CO (%)': np.random.uniform(0, 0.5, num_samples),
    'HC (ppm)': np.random.uniform(0, 100, num_samples),
    'CO2 (%)': np.random.uniform(10, 15, num_samples),
    'O2 (%)': np.random.uniform(0, 20, num_samples),
    'Lambda': np.random.uniform(0.8, 1.2, num_samples),
    'AFR': np.random.uniform(10, 20, num_samples)
}

# Create DataFrame from the generated data
df = pd.DataFrame(data)

# Compute the correlation matrix
corr_matrix = df.corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Engine Variables')
plt.show()
