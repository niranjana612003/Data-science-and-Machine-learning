import matplotlib.pyplot as plt
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
expenses = [2000, 2200, 2100, 2500, 2400]
plt.fill_between(months, expenses, color='skyblue', alpha=0.5)
plt.plot(months, expenses, color='blue')
plt.title("Monthly Expenses")
plt.xlabel("Month")
plt.ylabel("Expense (INR)")
plt.show()
plt.savefig("Monthly Expenses.png")