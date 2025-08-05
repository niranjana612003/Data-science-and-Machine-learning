import pandas as pd
df = pd.read_csv('employees.csv')
high_salary = df[df['Salary'] > 50000]
print(high_salary)