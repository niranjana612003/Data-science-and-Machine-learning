import matplotlib.pyplot as plt
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
productA = [200, 240, 210, 300, 280]
plt.bar(months, productA, color='blue')
plt.title("Product A Sales Over 5 Months")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()
plt.savefig("Product A Sales Over 5 Months.png")