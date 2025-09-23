import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Load your CSV dataset
data = pd.read_csv('/home/pc01/PycharmProjects/PythonProject5/flowers (1).csv')

# Split features and target
X = data.drop('species', axis=1)
y = data['species']

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Precision, Recall, F1-Score (Safe evaluation)
labels_present = np.unique(y_test)
target_names_present = le.inverse_transform(labels_present)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=labels_present,
    target_names=target_names_present
))