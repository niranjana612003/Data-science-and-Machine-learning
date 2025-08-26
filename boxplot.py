import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

file_path = "knn_sample_data.csv"
df = pd.read_csv(file_path)


salary_stats = df['Salary'].describe()

Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


print("Salary Descriptive Statistics:")
print(salary_stats)
print(f"Lower Bound for Outliers: {lower_bound}")
print(f"Upper Bound for Outliers: {upper_bound}")


plt.figure(figsize=(6,4))
sns.boxplot(x=df['Salary'])
plt.title("Box Plot of Salary")
plt.savefig('Box plot')
