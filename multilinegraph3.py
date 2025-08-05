import matplotlib.pyplot as plt
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
apple = [120, 125, 123, 130, 128, 132, 129]
google = [1000, 1005, 1010, 1020, 1015, 1030, 1025]
amazon = [1800, 1810, 1790, 1820, 1815, 1830, 1825]
plt.plot(days, apple, label='Apple')
plt.plot(days, google, label='Google')
plt.plot(days, amazon, label='Amazon')
plt.legend()
plt.title("Stock Prices Over a Week")
plt.xlabel("Day")
plt.ylabel("Price")
plt.show()
plt.savefig("Stock Prices Over a Week.png")