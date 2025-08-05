import matplotlib.pyplot as plt
activities = ['Sleep', 'Work', 'Exercise', 'Leisure', 'Others']
time = [8, 9, 1, 4, 2]
plt.pie(time, labels=activities, autopct='%1.1f%%')
plt.title("Daily Time Distribution")
plt.axis('equal')
plt.show()
plt.savefig("Daily Time Distribution.png")