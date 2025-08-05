from google.colab import files
uploaded = files.upload()
import numpy as np


data = np.genfromtxt('student.csv', delimiter=',', skip_header=1, usecols=2)
print("Scores:", data)

average = np.mean(data)
print("Average Score:", average)

