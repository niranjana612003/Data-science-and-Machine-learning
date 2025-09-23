import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Create dataset with numeric features and numeric target
np.random.seed(42)
X = pd.DataFrame({
    "Feature1": np.random.normal(0, 1, 500),
    "Feature2": np.random.normal(5, 2, 500)
})

# Target variable (numeric)
y = 2*X["Feature1"] - 3*X["Feature2"] + np.random.normal(0, 1, 500)

# Bin numeric target into 3 classes for classification
y_binned = pd.qcut(y, q=3, labels=[0,1,2])

X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.3, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy (on binned target): {accuracy_score(y_test, y_pred):.4f}")


X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y, test_size=0.3, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train_reg)
y_pred_reg = reg_model.predict(X_test)

mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"Regression MSE: {mse:.4f}")
print(f"Regression R2: {r2:.4f}")

