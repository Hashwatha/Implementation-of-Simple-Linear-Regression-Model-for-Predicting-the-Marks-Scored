#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')

#displaying the content in datafile
df.head()
df.tail()


# In[2]:


#segregating data to variables
X = df.iloc[:,:-1].values
print(X)


# In[3]:


Y=df.iloc[:,1].values
print(Y)


# In[4]:


#splitting train and test data

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)


# In[5]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)


# In[6]:


#displaying predicted values
print(Y_pred)


# In[7]:


#display actual values
print(Y_test)


# In[8]:


mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)


# In[9]:


#Graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


# In[10]:


#Graph plot for test data
plt.scatter(X_test,Y_test,color="pink")
plt.plot(X_test,Y_pred,color="black")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
print("Name:Hashwatha M")
print("Reg no:212223240051")


# In[ ]:




