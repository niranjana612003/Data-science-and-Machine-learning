import pandas as pd
df = pd.read_csv('employees.csv')
sorted_age = df.sort_values(by='Age')
print(sorted_age[['Name', 'Age']])