#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression of Iris dataset using Sklearn 

# In[51]:


#Import the dependencies 
import pandas as pd
import numpy as np
import scipy as sp
import sklearn
from sklearn import preprocessing
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set(style="darkgrid")


# In[52]:


#Load the data set 
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', names=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'])
df.head()


# In[53]:


df.info()


# # Preprocessing the data set

# In[54]:


#Check the null values from Iris Data set
df.isnull().sum()


# # Exploratory Data Analysis of the Iris Dataset

# In[55]:


#Counting the SepalLengthCm of Scpecies Iris- Setosa, Iris-Versicolor and Iris-Virginica.
plt.figure(figsize = (16,6))
sns.countplot(df['SepalLengthCm'], hue = df['Species'], palette = 'viridis')


# In[56]:


#Counting the SepalWidthCm of Scpecies Iris- Setosa, Iris-Versicolor and Iris-Virginica.
plt.figure(figsize = (16,6))
sns.countplot(df['SepalWidthCm'], hue = df['Species'], palette = 'plasma')


# In[57]:


#Counting the PetalLengthCm of Scpecies Iris- Setosa, Iris-Versicolor and Iris-Virginica.
plt.figure(figsize = (16,6))
sns.countplot(df['PetalLengthCm'], hue = df['Species'], palette = 'plasma')


# In[58]:


#Counting the PetalWidthCm of Scpecies Iris- Setosa, Iris-Versicolor and Iris-Virginica.
plt.figure(figsize = (16,6))
sns.countplot(df['PetalWidthCm'], hue = df['Species'], palette = 'inferno')


# In[59]:


#Relation between SepalWidthCm and SepalLengthCm based upon Species Iris- Setosa, Iris-Versicolor and Iris-Virginica.
plt.figure(figsize = (8,8))
sns.scatterplot(x = 'SepalLengthCm', y = 'SepalWidthCm' , data = df , hue = 'Species',
                palette = 'viridis' , s = 70)


# In[60]:


#Relation between PetalWidthCm and PetalLengthCm based upon Species Iris- Setosa, Iris-Versicolor and Iris-Virginica.
plt.figure(figsize = (8,8))
sns.scatterplot(x = 'PetalLengthCm', y = 'PetalWidthCm' , data = df , hue = 'Species', 
                palette = 'inferno' , s=70)


# In[61]:


#Shows the relation of the pair plot of SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm based on Species.
plt.figure(figsize = (8,8))
sns.pairplot(df)


# # Correlation Matrix of the Iris dataset

# In[62]:


#Shows the matrix Coorelation of the Iris database based on Species.
df.corr()


# In[63]:


correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='plasma')
plt.show()


# # Logistic Regression: Model Evaluation 

# In[64]:


x = df.drop(['Species'] , axis = 1)
y = df['Species']


# In[65]:


#Split the dataset into 70% training and 30% testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.30, random_state = 0)


# In[66]:


#Train the model 
from sklearn.linear_model import LogisticRegression
lreg = LogisticRegression()


# In[67]:


lreg.fit(x_train, y_train)
lreg.score(x_train,y_train)


# In[68]:


#Test the model 
predictions = lreg.predict(x_test)
predictions


# In[69]:


y_train


# # Comparison of the model based on accuracy

# In[70]:


from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
ypred = lreg.predict(x_test)

Report = confusion_matrix(y_test, ypred)
print(Report)


# In[71]:


Classic = classification_report(y_test, ypred)
print(Classic)


# # Logistic Regression: Label Encoder

# In[73]:


#Encodes the Species names into labels as 0,1,2.
from sklearn.preprocessing import LabelEncoder
LEC = LabelEncoder()


# In[74]:


df['Species']= LEC.fit_transform(df['Species'])
df.head()


# # Logistic Regression: Hyperparameter Tuning

# In[75]:


#Using Exhaustive Grid Search (GridSearchCV)

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

Pmeter = {'kernel':['linear', 'rbf', 'poly'], 'C':[0.1, 0.5, 1, 5, 10]}

clf = GridSearchCV(SVC(), Pmeter)
clf.fit(x_train, y_train)
print('score',clf.score(x_test, y_test))
print(clf.best_params_)


# # Logistic Regression: Accuracy

# In[76]:


#shows the accuracy of Iris data.
from sklearn.metrics import accuracy_score
print("Accuracy",accuracy_score(y_test, predictions)*100)


# In[ ]:




