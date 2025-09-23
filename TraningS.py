import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
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
}
df = pd.DataFrame(data)
df['House_Price'] = 5 * df['MedInc'] + 0.5 * df['OtherFeat1'] + np.random.normal(0, 20, size)

X = df.drop(columns='House_Price')
y = df['House_Price']

train_sizes = [0.5, 0.7, 0.9]

for train_size in train_sizes:
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)

    # Predict & score
    y_pred = lr.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)

    print(f"Training size: {train_size * 100:.0f}%, Test size: {(1 - train_size) * 100:.0f}%, R2 score: {r2:.3f}")
