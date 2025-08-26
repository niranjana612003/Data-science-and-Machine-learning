from sklearn . datasets import load_iris
import pandas as pd

iris = load_iris ()

df = pd . DataFrame ( data = iris . data , columns = iris . feature_names )
df ['target'] = iris . target
df ['species'] = df ['target'].apply ( lambda i : iris . target_names [ i ])

print ("\nFirst 10 rows of the Iris dataset :\n")
print ( df . head (10) )