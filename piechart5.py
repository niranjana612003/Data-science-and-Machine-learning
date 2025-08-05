import matplotlib.pyplot as plt

brands = ['Samsung', 'Apple', 'Xiaomi', 'OnePlus', 'Others']
share = [30, 25, 20, 10, 15]
plt.pie(share, labels=brands, autopct='%1.1f%%', startangle=140)
plt.title("Smartphone Market Share")
plt.axis('equal')
plt.show()
plt.savefig("Smartphone Market Share.png")