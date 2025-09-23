import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create dataset
np.random.seed(42)
size = 100
data = {
    'MedInc': np.random.normal(50, 10, size),
    'OtherFeat1': np.random.uniform(10, 100, size),
    'OtherFeat2': np.random.randint(0, 100, size),
    'NoiseFeat1': np.random.normal(0, 1, size),  # noise features to illustrate regularization
    'NoiseFeat2': np.random.normal(0, 1, size)
}
df = pd.DataFrame(data)
df['House_Price'] = 5 * df['MedInc'] + 0.5 * df['OtherFeat1'] + np.random.normal(0, 20, size)

X = df.drop(columns='House_Price')
y = df['House_Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (important for Ridge and Lasso)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
lr = LinearRegression()
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)

# Fit models
lr.fit(X_train_scaled, y_train)
ridge.fit(X_train_scaled, y_train)
lasso.fit(X_train_scaled, y_train)

# Predict
y_pred_lr = lr.predict(X_test_scaled)
y_pred_ridge = ridge.predict(X_test_scaled)
y_pred_lasso = lasso.predict(X_test_scaled)

# R2 scores
r2_lr = r2_score(y_test, y_pred_lr)
r2_ridge = r2_score(y_test, y_pred_ridge)
r2_lasso = r2_score(y_test, y_pred_lasso)

print(f"Linear Regression R2: {r2_lr:.3f}")
print(f"Ridge Regression R2: {r2_ridge:.3f}")
print(f"Lasso Regression R2: {r2_lasso:.3f}")

# Coefficients for each model
print("\nCoefficients:")
print("Linear Regression:", lr.coef_)
print("Ridge Regression:", ridge.coef_)
print("Lasso Regression:", lasso.coef_)
