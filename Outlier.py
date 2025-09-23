import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Create example dataset
np.random.seed(42)
size = 100
data = {
    'MedInc': np.random.normal(50, 10, size),
    'AveRooms': np.random.normal(5, 2, size),  # average rooms per household
    'OtherFeat1': np.random.uniform(10, 100, size),
}
df = pd.DataFrame(data)
df['House_Price'] = 5 * df['MedInc'] + 2 * df['AveRooms'] + 0.5 * df['OtherFeat1'] + np.random.normal(0, 20, size)

# Add extreme outliers manually
df.loc[0, 'MedInc'] = 150   # extreme high outlier in MedInc
df.loc[1, 'AveRooms'] = 20  # extreme high outlier in AveRooms

# 1. Plot boxplots to identify outliers
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.boxplot(df['MedInc'])
plt.title('Boxplot of MedInc')
plt.subplot(1,2,2)
plt.boxplot(df['AveRooms'])
plt.title('Boxplot of AveRooms')
plt.savefig("Outlier")

# Function to remove outliers based on 1.5*IQR rule
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers from both features
df_clean = remove_outliers_iqr(df, 'MedInc')
df_clean = remove_outliers_iqr(df_clean, 'AveRooms')

print(f"Original dataset size: {len(df)}")
print(f"Dataset size after outlier removal: {len(df_clean)}")

# Split original data
X_orig = df.drop(columns='House_Price')
y_orig = df['House_Price']
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_orig, y_orig, test_size=0.2, random_state=42)

# Scale original
scaler_o = StandardScaler()
X_train_o_scaled = scaler_o.fit_transform(X_train_o)
X_test_o_scaled = scaler_o.transform(X_test_o)

# Train & evaluate on original
lr = LinearRegression()
lr.fit(X_train_o_scaled, y_train_o)
y_pred_o = lr.predict(X_test_o_scaled)
r2_orig = r2_score(y_test_o, y_pred_o)

# Split cleaned data
X_clean = df_clean.drop(columns='House_Price')
y_clean = df_clean['House_Price']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

# Scale cleaned
scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)

# Train & evaluate on cleaned data
lr_clean = LinearRegression()
lr_clean.fit(X_train_c_scaled, y_train_c)
y_pred_c = lr_clean.predict(X_test_c_scaled)
r2_clean = r2_score(y_test_c, y_pred_c)

print(f"\nR2 score before outlier removal: {r2_orig:.3f}")
print(f"R2 score after outlier removal: {r2_clean:.3f}")
