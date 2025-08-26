import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

file_path = "knn_sample_data.csv"
df = pd.read_csv(file_path)

plt.figure(figsize=(6,5))
sns.scatterplot(x='Age', y='Salary', hue='Purchased', data=df, palette='coolwarm')
plt.title("Age vs Salary by Purchase Status")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.savefig('Scatter')
