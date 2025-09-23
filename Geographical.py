import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create synthetic dataset with Latitude, Longitude, and House_Price
np.random.seed(42)
size = 200

data = {
    'Latitude': np.random.uniform(34.0, 36.0, size),   # e.g., somewhere in California
    'Longitude': np.random.uniform(-118.5, -117.0, size),
    'MedInc': np.random.normal(50, 10, size),
    'House_Price': np.zeros(size)
}

df = pd.DataFrame(data)

# Generate house prices influenced by location and median income
# For example, prices increase as latitude increases and median income increases
df['House_Price'] = (df['Latitude'] - 34) * 10000 + df['MedInc'] * 300 + np.random.normal(0, 10000, size)

# Scatter plot of locations color-coded by house price
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df['Longitude'], df['Latitude'], c=df['House_Price'], cmap='viridis', alpha=0.8)
plt.colorbar(scatter, label='House Price')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geographical Distribution of House Prices')
plt.grid(True)
plt.savefig("Geographical")
