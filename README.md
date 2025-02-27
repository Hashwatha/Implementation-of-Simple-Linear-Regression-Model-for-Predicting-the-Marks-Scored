# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Hashwatha M
RegisterNumber: 212223240051
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')

#displaying the content in datafile
df.head()
df.tail()
```
```
#segregating data to variables
X = df.iloc[:,:-1].values
print(X)
```
```
Y=df.iloc[:,1].values
print(Y)
```
```
#splitting train and test data

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
```
```
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
```
```
#displaying predicted values
print(Y_pred)
```
```
#display actual values
print(Y_test)
```
```
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```
```
#Graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```
#Graph plot for test data
plt.scatter(X_test,Y_test,color="pink")
plt.plot(X_test,Y_pred,color="black")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
## Output:
## Displaying the content in datafield
## head:
![image](https://github.com/user-attachments/assets/ce7a689d-e0ff-4c90-8e92-045c2ea9b597)
## tail:
![image](https://github.com/user-attachments/assets/5fe75bb7-f41f-48ad-8784-46f0ed678240)
## Segregating data to variables
![image](https://github.com/user-attachments/assets/1925267a-4d10-4572-af8f-1a01d5f87a79)
## Displaying predicted values
![image](https://github.com/user-attachments/assets/63a91a54-7c60-4bb9-bf53-bbd55138b9be)
## Displaying actual values
![image](https://github.com/user-attachments/assets/b5f84e76-33c4-4fbb-8627-ab7828cf3737)
## MSE MAE RMSE
![image](https://github.com/user-attachments/assets/5fb890fa-5555-48a6-a563-f19135959eb2)
## Graph plot for training data
![image](https://github.com/user-attachments/assets/a6febe0d-9b26-4b03-bda3-bae4a177994e)
## Graph plot for test data
![image](https://github.com/user-attachments/assets/9ab8375c-ac3b-47d2-b782-ab3d1f820db2)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
