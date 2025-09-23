import pandas as pd
import numpy as np

# Sample dataset creation
np.random.seed(42)
data = {
    'Feature_A': np.random.normal(loc=50, scale=5, size=20),   # Normal dist, mean=50, std=5
    'Feature_B': np.random.uniform(low=10, high=100, size=20), # Uniform dist between 10 and 100
    'Feature_C': np.random.randint(0, 100, size=20),           # Random integers 0-99
    'Feature_D': np.random.exponential(scale=2, size=20),      # Exponential dist, mean=2
    'Feature_E': np.random.normal(loc=0, scale=1, size=20)     # Normal dist, mean=0, std=1
}

X = pd.DataFrame(data)

# Display the first 10 rows
print("First 10 rows of the dataset:")
print(X.head(10))

# Compute basic statistics
desc_stats = X.describe().T
desc_stats['median'] = X.median()
print("\nBasic statistics for each feature:")
print(desc_stats[['mean', 'median', 'min', 'max']])

# Calculate standard deviation for variation
desc_stats['std'] = X.std()

# Find the feature with the highest variation
max_var_feature = desc_stats['std'].idxmax()
max_var_value = desc_stats['std'].max()

print(f"\nFeature with the highest variation: '{max_var_feature}' (std = {max_var_value:.3f})")
