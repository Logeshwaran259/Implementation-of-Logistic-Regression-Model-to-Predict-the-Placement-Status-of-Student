# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm


## Program:
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: LOGESHWARAN
RegisterNumber:212220040081
*/


```
import pandas as pd
data=pd.read_csv('/content/Placement_Data (1).csv')
print("Placement data")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
print("Salary data")
data1.head()

print("Checking the null() function")
data1.isnull().sum()

print("Data Duplicate")
data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
print("Print data")
data1

x=data1.iloc[:,:-1]
print("Data-status")
x

y=data1["status"]
print("data-status")
y


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print("y_prediction array")
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy value")
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print("Confusion array")

confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print("Classification report")
print(classification_report1)

print("Prediction of LR")
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```




## Output:




![image](https://github.com/Logeshwaran259/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/129398164/dff94f40-61e9-4a86-a5da-a546ca6e8546)





![image](https://github.com/Logeshwaran259/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/129398164/3e7fceea-5016-492c-85fc-208b52eaaac8)



![image](https://github.com/Logeshwaran259/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/129398164/2c238d75-ac2a-4998-a5cc-b6f081032b80)




![image](https://github.com/Logeshwaran259/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/129398164/8dd7d10d-1865-45f3-802d-099a7488fb56)





![image](https://github.com/Logeshwaran259/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/129398164/330412b9-a85e-499c-808e-a365567fb7f2)







![image](https://github.com/Logeshwaran259/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/129398164/4816e08d-d1ae-4e0f-b820-9d496ba01411)







![image](https://github.com/Logeshwaran259/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/129398164/384a0e09-2ab3-4ddc-baa7-16030b25e2be)







## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
