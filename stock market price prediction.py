#!/usr/bin/env python
# coding: utf-8

# In[4]:


pip install chart_studio


# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly.express as px

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# In[6]:


df = pd.read_csv('EWMAX.csv')


# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


df['Date'] = pd.to_datetime(df['Date'])


# In[10]:


print(f'Dataframe contains stock prices between {df.Date.min()} {df.Date.max()}')
print(f'Total days = {(df.Date.max()-df.Date.min()).days} days')


# In[11]:


df.describe()


# In[12]:


df[['Open', 'High', 'Low', 'Close']].plot(kind='box', subplots=True, figsize=(10,10), layout=(2,2))


# In[13]:


df_data = [{'x':df['Date'], 'y': df['Close']}]
plot = go.Figure(data=df_data)
plot.update_layout(title='Closing Price', xaxis_title='Date', yaxis_title='Closing Price')
plot.show()


# In[14]:


iplot(plot)


# In[15]:


pip install keras


# In[16]:


pip install scikit-learn


# In[17]:


get_ipython().system('pip install tensorflow')


# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score


# In[19]:


X = np.array(df.index).reshape(-1,1)
Y = df['Close']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=110)


# In[20]:


scaler = StandardScaler().fit(X_train)


# In[21]:


from sklearn.linear_model import LinearRegression


# In[22]:


lm= LinearRegression()
lm.fit(X_train, Y_train)


# In[23]:


trace0 = go.Scatter(x=X_train.T[0], y=Y_train, mode='markers', name='Training Data')
trace1 = go.Scatter(x=X_test.T[0], y=Y_test, mode='markers', name='Testing Data')


df_data = [trace0, trace1]
plot = go.Figure(data=df_data)
plot.update_layout(title='Closing Price', xaxis_title='Date', yaxis_title='Closing Price')
plot.show()


# In[24]:


iplot(plot)


# In[25]:


scores=f'''
{'Metric'.ljust(10)}{'Train'.center(20)}{'Test'.center(20)}
{'r2_score'.ljust(10)}{r2_score(Y_train, lm.predict(X_train))}\t{r2_score(Y_test,lm.predict(X_test))}
{'MSE'.ljust(10)}{mse(Y_train, lm.predict(X_train))}\t{mse(Y_test,lm.predict(X_test))}
'''
print(scores)


# In[ ]:




