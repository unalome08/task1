#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"DATA SCIENCE AND BUSINESS ANALYTICS"

Prediction using Supervised ML #task 1

"Predict the percentage of an student based on the no. of study hours."

GRIP AUGUST21 BATCH

NAME:"PRATIBHA VISHWAKARMA"


# In[51]:


#installing dependencies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


import warnings 
warnings.filterwarnings('ignore')


# In[9]:


#importing the data
data =pd.read_csv("http://bit.ly/w-data")
print('data imported successfully !')
data.head(5)


# In[12]:


data.shape  #has 25 rows and 2 columns


# In[14]:


data.columns


# In[ ]:


#cleaning the data


# In[235]:


data.isnull().sum() #has no nullvalue


# In[ ]:


#graph between hours and scores


# In[240]:


data.plot(x='Hours' , y='Scores', style='o',color='red')
plt.title('HOURS V/S SCORE')
plt.xlabel('hours(hrs)')
plt.ylabel('scores(%)')


# In[ ]:


# model training,testing and splitting the model


# In[200]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X=np.array(data['Hours']).reshape((-1,1))
y=np.array(data['Scores'])


# In[201]:


X_train,x_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)


# In[202]:


regr = LinearRegression()
regr.fit(X_train, y_train)


# In[ ]:


#plotting a regression graph:


# In[226]:


plt.scatter(X,y)
plt.title('HOURS V/S SCORE')
y_prd=regr.predict(X)
plt.plot(X,y_prd,color='red')
plt.xlabel('hours(hrs)')
plt.ylabel('scores(%)')
plt.show()


# In[236]:


#prediction of scores a student get after studying 9.25 hours/day
hours=9.25
my_pred=regr.predict([[hours]])
print("no.of hours studied {}".format(hours))
print("students predicted percentage{}".format(my_pred[0]))


# In[230]:


from sklearn import metrics
print('mean absolute error',metrics.mean_absolute_error(y,y_prd))


# In[ ]:


#1 Task completed ! Thank you.
