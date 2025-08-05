import pandas as pd
df = pd.read_csv('employees.csv')
hr_employees = df[df['Department'] == 'HR']['Name']
print(hr_employees)