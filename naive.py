import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("/home/pc01/Downloads/play_tennis_dataset_500.csv")

# Separate features and target
X = df.drop("PlayTennis", axis=1)
y = df["PlayTennis"]

# Encode categorical columns
encoders = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Naive Bayes model for categorical data
model = CategoricalNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
probs = model.predict_proba(X_test)
# Example: predict class 1 if probability > 0.6 instead of 0.5
y_pred_custom = (probs[:, 1] > 0.6).astype(int)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred_custom)
precision = precision_score(y_test, y_pred_custom, average="weighted")
recall = recall_score(y_test, y_pred_custom, average="weighted")
f1 = f1_score(y_test, y_pred_custom, average="weighted")

# Print neatly
print("Model Performance Metrics:")
print(f"Accuracy   : {accuracy:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"F1-Score   : {f1:.4f}")

