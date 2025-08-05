import pandas as pd
df = pd.read_csv('employees.csv')
import matplotlib.pyplot as plt
plt.bar(df['Name'], df['Salary'], color='skyblue')
plt.title('Employee Salaries')  # Fixed the quote
plt.xlabel('Employee Name')
plt.ylabel('Salary (INR)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.savefig('emplyee.png')

