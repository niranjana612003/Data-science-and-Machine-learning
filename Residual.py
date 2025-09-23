import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Create example dataset with feature 'MedInc' and target 'House_Price'
np.random.seed(42)
size = 100

data = {
    'MedInc': np.random.normal(50, 10, size),       # Median Income
    'OtherFeat1': np.random.uniform(10, 100, size),
    'OtherFeat2': np.random.randint(0, 100, size),
}

df = pd.DataFrame(data)

# Target depends mostly on MedInc plus some noise and influence from other features
df['House_Price'] = 5 * df['MedInc'] + 0.5 * df['OtherFeat1'] + np.random.normal(0, 20, size)

# Split into train and test sets
X = df.drop(columns='House_Price')
y = df['House_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simple Linear Regression using 'MedInc'
lr_simple = LinearRegression()
lr_simple.fit(X_train[['MedInc']], y_train)

# Predict on test set
y_pred_simple = lr_simple.predict(X_test[['MedInc']])

# Calculate R2 score
r2_simple = r2_score(y_test, y_pred_simple)
print(f"R2 score (Simple Linear Regression with MedInc): {r2_simple:.3f}")

# Calculate residuals
residuals = y_test - y_pred_simple

# Plot residuals
plt.scatter(y_pred_simple, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted House_Price')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot for Simple Linear Regression')
plt.savefig("Residual")
