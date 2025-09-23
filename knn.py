from sklearn . datasets import load_iris
from sklearn . model_selection import train_test_split
from sklearn . neighbors import KNeighborsClassifier
from sklearn . metrics import accuracy_score
import pandas as pd
# Step 2: Load the Iris dataset
iris = load_iris ()
# Step 5: Define features (X) and target (y)
X = iris . data # Feature columns
y = iris . target # Class labels : 0 , 1 , 2
# Step 6: Split into training (80%) and testing (20%) sets
X_train , X_test , y_train , y_test = train_test_split (
X , y , test_size =0.2 , random_state =1
)
# Step 7: Initialize and train the k-NN classifier (k =10)
knn = KNeighborsClassifier ( n_neighbors =10)
knn . fit ( X_train , y_train )
# Step 8: Predict the test data and calculate accuracy
y_pred = knn . predict ( X_test )
accuracy = accuracy_score ( y_test , y_pred )
print ( f" Model Accuracy on Test Data : { accuracy * 100:.2f}%\n")
# Step 9: Predict species for a new flower sample
sample = [[6.0 , 2.9 , 4.5 , 1.5]] # Example : new flower
predicted_class = knn . predict ( sample ) [0]
predicted_name = iris . target_names [ predicted_class ]
print ( f" Sample Input : { sample }")
print ( f" Predicted Class Index : { predicted_class }")
print ( f" Predicted Class Name : { predicted_name }")