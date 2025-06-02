# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset.
2. Preprocess the data(eg., handle categorical variables like salary).
3. Select relevant features and the target variables.
4. Split the data into training and the target variable.
5. Train the Desicion Tree Classifier on the training data.
6. Predict the outcomes using the test data
7. Evaluate the midel's accuracy.
8. Visualize the Decision Tree.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Reshma R
RegisterNumber:  212224040274
*/
```
```
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

data=pd.read_csv("/content/Employee.csv")

le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])

features = ["satisfaction_level","last_evaluation","number_project","average_montly_hours","salary"]
print(data.columns)
x=data[features]
y=data["left"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)
accuracy=metrics.accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

plt.figure(figsize=(12,8))
plot_tree(dt,feature_names=features,class_names=["Not Left","Left"],filled=True)
plt.show()
```
## Output:
![Screenshot 2025-04-22 194522](https://github.com/user-attachments/assets/a6ede654-7cd3-4aeb-9a69-d165e8efb1e2)
![Screenshot 2025-04-22 194532](https://github.com/user-attachments/assets/986febcd-eebe-46aa-9592-faf8a2f3fc51)


![Screenshot 2025-04-22 194538](https://github.com/user-attachments/assets/8f1a598d-f10c-4fba-87de-302d82e92133)
![Screenshot 2025-04-22 194613](https://github.com/user-attachments/assets/c455f21d-c85c-482d-b7ca-6687db020a4e)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
