pip install chart_studio


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly.express as px

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)





df = pd.read_csv('EWMAX.csv')





df.head()





df.info()





df['Date'] = pd.to_datetime(df['Date'])




print(f'Dataframe contains stock prices between {df.Date.min()} {df.Date.max()}')
print(f'Total days = {(df.Date.max()-df.Date.min()).days} days')




df.describe()





df[['Open', 'High', 'Low', 'Close']].plot(kind='box', subplots=True, figsize=(10,10), layout=(2,2))




df_data = [{'x':df['Date'], 'y': df['Close']}]
plot = go.Figure(data=df_data)
plot.update_layout(title='Closing Price', xaxis_title='Date', yaxis_title='Closing Price')
plot.show()





iplot(plot)





pip install keras





pip install scikit-learn





get_ipython().system('pip install tensorflow')





from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score




X = np.array(df.index).reshape(-1,1)
Y = df['Close']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=110)





scaler = StandardScaler().fit(X_train)





from sklearn.linear_model import LinearRegression





lm= LinearRegression()
lm.fit(X_train, Y_train)





trace0 = go.Scatter(x=X_train.T[0], y=Y_train, mode='markers', name='Training Data')
trace1 = go.Scatter(x=X_test.T[0], y=Y_test, mode='markers', name='Testing Data')


df_data = [trace0, trace1]
plot = go.Figure(data=df_data)
plot.update_layout(title='Closing Price', xaxis_title='Date', yaxis_title='Closing Price')
plot.show()





iplot(plot)




scores=f'''
{'Metric'.ljust(10)}{'Train'.center(20)}{'Test'.center(20)}
{'r2_score'.ljust(10)}{r2_score(Y_train, lm.predict(X_train))}\t{r2_score(Y_test,lm.predict(X_test))}
{'MSE'.ljust(10)}{mse(Y_train, lm.predict(X_train))}\t{mse(Y_test,lm.predict(X_test))}
'''
print(scores)







