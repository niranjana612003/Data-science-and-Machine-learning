import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

df=pd.read_csv('iris_dataset.csv')
print(df.head())
print(df.isnull().sum())

x=df[['sepallength','sepalwidth','petallength','petalwidth']]
y=df['species']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train,y_train)

y_pred=knn.predict(x_test)

print("Accuracy Score: ",accuracy_score(y_test,y_pred))
print("Confusion matrix: ",confusion_matrix(y_test,y_pred))
print("classification matrix: ",classification_report(y_test,y_pred))


#plt.figure()
plt.scatter(y_test,y_pred,s=20)
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color="red")
plt.title("IRIS visualization")
plt.xlabel("actual values")
plt.ylabel("predicted values")
plt.savefig("IRIS")