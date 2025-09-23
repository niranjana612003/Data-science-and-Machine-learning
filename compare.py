import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Load dataset
data = pd.read_csv('/home/pc01/PycharmProjects/PythonProject5/flowers (1).csv')

# Features and target
X = data.drop('species', axis=1)
y = data['species']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Initialize models
dt_clf = DecisionTreeClassifier(random_state=42)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train models
dt_clf.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)

# Predict
dt_pred = dt_clf.predict(X_test)
rf_pred = rf_clf.predict(X_test)

# Get present labels in test set
labels_present = np.unique(y_test)
target_names_present = le.inverse_transform(labels_present)

# Print Decision Tree results
print("Decision Tree Classifier Results")
print("-------------------------------")
print(f"Accuracy: {accuracy_score(y_test, dt_pred):.2f}")
print(classification_report(y_test, dt_pred, labels=labels_present, target_names=target_names_present))

# Print Random Forest results
print("Random Forest Classifier Results")
print("-------------------------------")
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.2f}")
print(classification_report(y_test, rf_pred, labels=labels_present, target_names=target_names_present))
