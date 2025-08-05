import matplotlib.pyplot as plt

days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
temps = [30, 32, 31, 29, 28, 27, 26]

plt.plot(days, temps, marker='o')
plt.title("Weekly Temperatures")
plt.xlabel("Days")
plt.ylabel("Temperature (°C)")
plt.grid()
plt.show()
plt.savefig("weekly_temperatures.png")