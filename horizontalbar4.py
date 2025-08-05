import matplotlib.pyplot as plt

cities = ['City A', 'City B', 'City C', 'City D']
population = [150000, 120000, 180000, 100000]
plt.barh(cities, population, color='orange')
plt.title("Population by City")
plt.xlabel("Population")
plt.ylabel("City")
plt.show()
import matplotlib.pyplot as plt