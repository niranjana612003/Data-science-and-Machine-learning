import numpy as np

marks = np.random.randint(0, 101, size=100)
print("Marks:\n", marks)

average = np.mean(marks)

highest = np.max(marks)

lowest = np.min(marks)

pass_count = np.sum(marks >= 50)
pass_percentage = (pass_count / len(marks)) * 100


print(f"\nAverage Marks: {average:.2f}")
print(f"Highest Marks: {highest}")
print(f"Lowest Marks: {lowest}")
print(f"Pass Percentage: {pass_percentage:.2f}%")
