#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""1.)Import necessary libraries: pandas, NumPy, matplotlib, seaborn, and sklearn.
2.)Load the dataset from the given URL using pandas.
3.Perform exploratory data analysis to get an overview of the dataset.
4.Handle missing values by dropping the rows with missing values.
5.Select the features and target variables.
6.Split the dataset into training and testing sets using the train_test_split() function from sklearn.
7.Build a logistic regression model using the LogisticRegression() function from sklearn.
8.Evaluate the model using classification_report() function from sklearn.
9.Plot the distribution of the target variable using seaborn's countplot() function.
""""""


# In[43]:


# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# In[9]:


# Loading the dataset
df = pd.read_csv("https://raw.githubusercontent.com/Premalatha-success/Datasets/main/h1n1_vaccine_prediction.csv")


# In[15]:


df.sample(12)


# In[10]:


# Exploratory Data Analysis
print(df.head())
print(df.describe())


# In[11]:


# Handling missing values
print(df.isna().sum())
df = df.dropna()


# In[39]:


# Encoding categorical variables
le = LabelEncoder()
df["age_bracket"] = le.fit_transform(df["age_bracket"])
df["qualification"] = le.fit_transform(df["qualification"])
df["race"] = le.fit_transform(df["race"])
df["sex"] = le.fit_transform(df["sex"])
df["income_level"] = le.fit_transform(df["income_level"])
df["marital_status"] = le.fit_transform(df["marital_status"])
df["housing_status"] = le.fit_transform(df["housing_status"])
df["employment"] = le.fit_transform(df["employment"])
df["census_msa"] = le.fit_transform(df["census_msa"])


# In[40]:


# Feature selection
X = df.drop(["h1n1_vaccine"], axis=1)
y = df["h1n1_vaccine"]


# In[41]:


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[44]:


# create a StandardScaler object
scaler = StandardScaler()


# In[45]:


# fit and transform the training data
X_train = scaler.fit_transform(X_train)


# In[46]:


# transform the test data
X_test = scaler.transform(X_test)


# In[47]:


# create a Logistic Regression object and fit the model
lr = LogisticRegression()
lr.fit(X_train, y_train)


# In[48]:


# Predict on the test data
y_pred = lr.predict(X_test)


# In[49]:


# Model building
model = LogisticRegression()
model.fit(X_train, y_train)


# In[50]:


# Model evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# In[58]:


# Plotting the results
sns.countplot(x="h1n1_vaccine", data=df)
plt.show()


# In[ ]:




