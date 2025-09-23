import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score

# Create dataset
np.random.seed(42)
size = 100
data = {
    'MedInc': np.random.normal(50, 10, size),
    'OtherFeat1': np.random.uniform(10, 100, size),
    'OtherFeat2': np.random.randint(0, 100, size),
}
df = pd.DataFrame(data)
df['House_Price'] = 5 * df['MedInc'] + 0.5 * df['OtherFeat1'] + np.random.normal(0, 20, size)

X = df.drop(columns='House_Price')
y = df['House_Price']

# Scale features before CV (fit_transform inside CV will cause data leakage)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define model
lr = LinearRegression()

# Define KFold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation and get R2 scores
r2_scores = cross_val_score(lr, X_scaled, y, cv=kf, scoring='r2')

print(f"R2 scores for each fold: {r2_scores}")
print(f"Mean R2 score: {r2_scores.mean():.3f}")
print(f"Standard deviation of R2 scores: {r2_scores.std():.3f}")
