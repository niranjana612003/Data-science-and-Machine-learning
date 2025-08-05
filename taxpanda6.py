import pandas as pd
df = pd.read_csv('employees.csv')
df['Tax'] = df['Salary'] * 0.10
print(df[['Name', 'Salary', 'Tax']])