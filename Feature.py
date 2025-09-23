import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Create example dataset
np.random.seed(42)
size = 100
data = {
    'MedInc': np.random.normal(50, 10, size),
    'OtherFeat1': np.random.uniform(10, 100, size),
    'OtherFeat2': np.random.randint(0, 100, size),
}
df = pd.DataFrame(data)
df['House_Price'] = 5 * df['MedInc'] + 0.5 * df['OtherFeat1'] + np.random.normal(0, 20, size)

# Split dataset
X = df.drop(columns='House_Price')
y = df['House_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Without scaling ---
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
r2_no_scale = r2_score(y_test, y_pred)
coeff_no_scale = lr.coef_

print("Without Scaling:")
print(f"R2 score: {r2_no_scale:.3f}")
print(f"Coefficients: {coeff_no_scale}")

# --- With StandardScaler ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_scaled = LinearRegression()
lr_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = lr_scaled.predict(X_test_scaled)
r2_scaled = r2_score(y_test, y_pred_scaled)
coeff_scaled = lr_scaled.coef_

print("\nWith StandardScaler:")
print(f"R2 score: {r2_scaled:.3f}")
print(f"Coefficients: {coeff_scaled}")
