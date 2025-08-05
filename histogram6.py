import matplotlib.pyplot as plt

scores = [56, 67, 45, 88, 72, 90, 61, 76, 84, 69]
plt.hist(scores, bins=5, color='green', edgecolor='black')
plt.title("Score Distribution")
plt.xlabel("Scores")
plt.ylabel("Number of Students")
plt.show()
plt.savefig("Score Distribution.png")