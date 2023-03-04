#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from geopy import distance


# In[6]:


# Load the data into a Pandas DataFrame
df = pd.read_csv('https://raw.githubusercontent.com/Premalatha-success/Datasets/main/TaxiFare.csv',encoding='utf-8')


# In[ ]:


#show the first few rows of the DataFrame
df.head()


# In[9]:


df.dtypes


# In[10]:


# Check for missing values
print(df.isnull().sum())


# In[11]:


# Remove any rows with missing values
df.dropna(inplace=True)


# In[12]:


# Convert the pickup and dropoff coordinates to (lat, long) tuples
pickup_coords = df[['longitude_of_pickup','latitude_of_pickup']].values
dropoff_coords = df[['longitude_of_dropoff','latitude_of_dropoff']].values


# In[13]:


# Calculate the distance between pickup and dropoff points using geopy's distance function
distances = [distance.distance(pickup_coords[i], dropoff_coords[i]).km for i in range(len(df))]
df.loc[:, 'distance'] = distances


# In[14]:


# Convert pickup_datetime column to datetime type and extract datetime features
df.loc[:, 'date_time_of_pickup'] = pd.to_datetime(df['date_time_of_pickup'])
df.loc[:, 'hour'] = df['date_time_of_pickup'].dt.hour
df.loc[:, 'day'] = df['date_time_of_pickup'].dt.day
df.loc[:, 'month'] = df['date_time_of_pickup'].dt.month


# In[15]:


# Convert relevant columns to numeric data types
df = df.apply(pd.to_numeric, errors='coerce', downcast='float')


# In[16]:


df.dtypes


# In[17]:


# Remove any rows with fare_amount <= 0
df = df.loc[df['amount'] > 0]


# In[18]:


# Plot the distribution of the amount column
sns.displot(df['amount'])


# In[19]:


# Remove any rows with amount <= 0
df = df[df['amount'] > 0]


# In[20]:


df.drop('unique_id', axis=1)
df.dtypes


# In[29]:


df.fillna(0, inplace=True)  # fill missing values with 0


# In[30]:


# Split the data into training and testing sets
X = df.drop('amount', axis=1)
y = df['amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[39]:


#For rescalling the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)


# In[35]:


# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# In[36]:


# Make predictions on the testing set
y_pred = model.predict(X_test)


# In[37]:


# Compute the root mean squared error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)


# In[38]:


# Plot the predicted vs. actual fares
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.show()


# In[ ]:




