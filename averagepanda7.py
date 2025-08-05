import pandas as pd
df = pd.read_csv('employees.csv')
avg_salary = df.groupby('Department')['Salary'].mean()
print(avg_salary)