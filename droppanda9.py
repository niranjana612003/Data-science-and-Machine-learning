import pandas as pd
df = pd.read_csv('employees.csv')
cleaned_df = df.dropna()
print(cleaned_df)