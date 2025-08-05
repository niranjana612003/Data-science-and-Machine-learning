import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('employees.csv')
department_counts = df['Department'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(
    department_counts,
    labels=department_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops={'edgecolor': 'white'}
)
plt.title('Department-wise Employee Distribution')
plt.axis('equal')
plt.tight_layout()
plt.show()
plt.savefig('peichart.png')


