import matplotlib.pyplot as plt
x = [10, 20, 30, 40, 50]
y = [15, 25, 35, 45, 55]
sizes = [100, 300, 500, 700, 900]
plt.scatter(x, y, s=sizes, alpha=0.5, c='purple')
plt.title("Bubble Chart Example")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.show()
plt.savefig("Bubble Chart Example.png")