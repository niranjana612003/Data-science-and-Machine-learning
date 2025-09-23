import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Sample dataset creation (features + target variable 'House_Price')
np.random.seed(42)
data = {
    'Feature_A': np.random.normal(loc=50, scale=5, size=20),
    'Feature_B': np.random.uniform(low=10, high=100, size=20),
    'Feature_C': np.random.randint(0, 100, size=20),
    'Feature_D': np.random.exponential(scale=2, size=20),
    'Feature_E': np.random.normal(loc=0, scale=1, size=20),
}

X = pd.DataFrame(data)

# Create a synthetic target variable correlated with some features
# For example, house price depends mostly on Feature_A and Feature_B plus noise
noise = np.random.normal(0, 10, size=20)
X['House_Price'] = 3*X['Feature_A'] + 2*X['Feature_B'] + noise

# Display first 10 rows
print("First 10 rows:")
print(X.head(10))

# Compute and print basic stats
desc_stats = X.describe().T
desc_stats['median'] = X.median()
print("\nBasic statistics:")
print(desc_stats[['mean', 'median', 'min', 'max']])

# Correlation matrix including target
corr_matrix = X.corr()

# Plot heatmap of correlations
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig("Correlation")

# Find feature with strongest correlation to 'House_Price' (excluding 'House_Price' itself)
corr_target = corr_matrix['House_Price'].drop('House_Price')
strongest_feature = corr_target.abs().idxmax()
strongest_corr_value = corr_target[strongest_feature]

print(f"\nFeature with the strongest correlation to House_Price: '{strongest_feature}' (correlation = {strongest_corr_value:.2f})")
