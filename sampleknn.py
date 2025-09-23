import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report


df = pd.read_csv('/home/pc01/Downloads/knn_sample_data.csv')
print(df.head())
print(df[['Age', 'Salary', 'Purchased']].dtypes)
print(df.isnull().sum())
print(df['Salary'].describe())

sns.boxplot(x=df['Salary'])
plt.title('Boxplot of Salary')
plt.savefig("me")

df[['Salary', 'Age']].hist(bins=15, figsize=(14,8))
plt.suptitle('Histograms of Age and Salary')
plt.savefig("meme")

sns.scatterplot(data=df, x='Age', y='Salary', hue='Purchased', palette='Set1')
plt.title('Age vs Salary by Purchased Status')
plt.savefig("meeeeee")

X = df[['Age', 'Salary']]
y = df['Purchased']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Instantiate KNN classifier with k=10
knn = KNeighborsClassifier(n_neighbors=10)

print(f"Set n_neighbors to: {knn.n_neighbors}")

# Train the model
knn.fit(X_train_scaled, y_train)

accuracy = knn.score(X_test_scaled, y_test)
print(f"Accuracy on test set: {accuracy:.2f}")

y_pred = knn.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

print(classification_report(y_test, y_pred))