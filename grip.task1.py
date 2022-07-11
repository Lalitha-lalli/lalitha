#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation GRIP #JUNE27 -- Data Science and Business Analytics Internship
# Task - 1 Prediction using supervised Machine Learning
# 
# By velugu lalitha
# 
# Problem Statement
# 
# 1.To predict the percentage of a student based on the number of study hours
# 
# 2.Predict the score of a student studies for 9.25 hrs/day
# 
# 
# 
# 

# In[1]:


# required modules importing
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


## reading the dataset using pandas
data=pd.read_csv('http://bit.ly/w-data')
data.head()


# In[4]:


# now we are knowing the shape ( No of rows, No of cols)
data.shape


# In[5]:


#  giving  the information and describes the data 
data.info()
data.describe()


# In[6]:


# plotting  the data 
sns.scatterplot(x=data['Hours'],y=data['Scores'])


# In[7]:


#lets do the regression  to get better understanding
sns.regplot(x=data['Hours'],y=data['Scores'],color='red');


# ## Lets prepare the data after the regression process:

# In[32]:


#Dividing the data into inputs and outputs
x=data[['Hours']]
y=data['Scores']


# In[33]:


#Splitting the data for both training and testing
from sklearn.model_selection import train_test_split
train_x, val_x, train_y, val_y = train_test_split(x,y, random_state=0)


# In[34]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# # Training the model:

# In[36]:


regressor.fit(train_x,train_y)


# # Predicting the data:

# In[38]:


pred_y = regressor.predict(val_x)


# In[39]:


pd.DataFrame({'Actual': val_y, 'Predicted':pred_y})


# In[40]:


## Actual vs predicted distribution plot graph
sns.kdeplot(pred_y, label="Predicted", shade=True);
sns.kdeplot(data=val_y, label="Actual", shade=True);


# In[41]:


# Train and test accuracy
print("training Accuracy is:", regressor.score(train_x, train_y),"\n test accuracy is:", regressor.score(val_x,val_y))


# In[42]:


## Mean Absolute error
from sklearn import metrics
print('Mean absolute Error:', metrics.mean_absolute_error(val_y, pred_y))
print("Max Error:", metrics.max_error(val_y, pred_y))


# In[43]:


# Here is the predicted score, if a student studies for a 9.25 hrs / day
h=[[9.25]]
s=regressor.predict(h)
print('A student who studies ', h[0][0], 'hours is estimated to score ', s[0])


# # The final output is:

# In[ ]:


If a student who studies 9.25 hrs/day is estimated to score 93.89272889341655

