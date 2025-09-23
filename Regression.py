import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Creating example dataset with a feature 'MedInc' and target 'House_Price'
np.random.seed(42)
size = 100

data = {
    'MedInc': np.random.normal(50, 10, size),       # Median Income
    'OtherFeat1': np.random.uniform(10, 100, size),
    'OtherFeat2': np.random.randint(0, 100, size),
    'House_Price': None
}

df = pd.DataFrame(data)
# Make House_Price depend mostly on MedInc + some noise + influence from other features
df['House_Price'] = 5 * df['MedInc'] + 0.5 * df['OtherFeat1'] + np.random.normal(0, 20, size)

# Split data into train and test sets
X = df.drop(columns='House_Price')
y = df['House_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simple Linear Regression with single feature MedInc
lr_simple = LinearRegression()
lr_simple.fit(X_train[['MedInc']], y_train)
y_pred_simple = lr_simple.predict(X_test[['MedInc']])
r2_simple = r2_score(y_test, y_pred_simple)

# Multiple Linear Regression with all features
lr_multi = LinearRegression()
lr_multi.fit(X_train, y_train)
y_pred_multi = lr_multi.predict(X_test)
r2_multi = r2_score(y_test, y_pred_multi)

# Print R2 scores
print(f"R2 score (Simple Linear Regression with MedInc): {r2_simple:.3f}")
print(f"R2 score (Multiple Linear Regression with all features): {r2_multi:.3f}")

# Plot regression line for simple model
plt.scatter(X_test['MedInc'], y_test, color='blue', label='Actual')
plt.plot(X_test['MedInc'], y_pred_simple, color='red', label='Regression Line')
plt.xlabel('MedInc')
plt.ylabel('House_Price')
plt.title('Simple Linear Regression: MedInc vs House_Price')
plt.legend()
plt.savefig("Regression")
