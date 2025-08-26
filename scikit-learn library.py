from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

file_path = "knn_sample_data.csv"
df = pd.read_csv(file_path)

knn = KNeighborsClassifier()

print(knn)
