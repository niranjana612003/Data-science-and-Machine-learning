import pandas as pd

file_path = "knn_sample_data.csv"
df = pd.read_csv(file_path)

from sklearn.model_selection import train_test_split


X = df[['Age', 'Salary']]
y = df['Purchased']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set size:", X_train.shape[0])
print("Testing set size:", X_test.shape[0])
