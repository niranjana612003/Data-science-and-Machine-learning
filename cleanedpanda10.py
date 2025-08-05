import pandas as pd

from droppanda9 import cleaned_df

df = pd.read_csv('employees.csv')
cleaned_df.to_csv('cleaned_employees.csv', index=False)
print(cleaned_df)