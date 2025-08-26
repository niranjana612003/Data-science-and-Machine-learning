import matplotlib.pyplot as plt
import pandas as pd
file_path = "knn_sample_data.csv"
df = pd.read_csv(file_path)

plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
plt.hist(df['Age'], bins=15, color='skyblue', edgecolor='black')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")


plt.subplot(1, 2, 2)
plt.hist(df['Salary'], bins=15, color='salmon', edgecolor='black')
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig('Histogram')
