import numpy as np

scores = np.array([85, 90, 78, 92, 88, 76, 95, 89])

mean = np.mean(scores)
median = np.median(scores)
std_dev = np.std(scores)
variance = np.var(scores)

print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Standard Deviation: {std_dev}")
print(f"Variance: {variance}")