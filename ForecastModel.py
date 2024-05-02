#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import warnings
from sklearn.metrics import mean_squared_error
from math import sqrt

warnings.filterwarnings("ignore")
def ad_test(data):
    dftest = adfuller(data,autolag = 'AIC')
    print("1. ADf:",dftest[0])
    print("2. p-value:",dftest[1])
    print("3. Num of Lags:",dftest[2])
    print("4. Num of observations used:",dftest[3])
    print("5. Critical Values:",dftest[3])
    for key,val in dftest[4].items():
        print(key,":",val)
        
        
# Load the data
data = pd.read_csv("ForecastData.csv", index_col="Month", parse_dates=True)
print(data)
ad_test(data)
# Fit ARIMA model
#model = ARIMA(data, order=(p, d, q))
#model_fit = model.fit()

# Forecast temperatures for Jan-24 to Dec-24
#forecast = model_fit.forecast(steps=12)
#print(forecast)
stepwise_fit  = auto_arima(data['Temperature'],trace = True,supress_warnings = True)
stepwise_fit.summary()

train = data.iloc[:-30]
test = data.iloc[-30:]
print(train.shape,test.shape)


# In[2]:


train = data.iloc[:-30]
test = data.iloc[-30:]
print(train.shape,test.shape)

model = ARIMA(train['Temperature'],order = (2,0,3))
model = model.fit()
model.summary()


# In[7]:


start = len(train)
end = len(train) +len(test) -1 
pred = model.predict(start = start,end = end,type = 'levels')
pred = pd.DataFrame(pred,columns = ['predicted_mean'])
predicted_index = pd.date_range(start='2021-07-01', periods=len(pred), freq='M')
predicted_index_str = predicted_index.strftime('%b-%y')

pred['predicted_index'] = predicted_index_str
pred.set_index('predicted_index', inplace=True)

print(pred)

print(test['Temperature'])
pred.plot(legend=True,label='Temperature predicted by model')
test['Temperature'].plot(legend = True)


# In[4]:


print(test['Temperature'].mean())
rmse = sqrt(mean_squared_error(pred,test['Temperature']))
print(rmse)


# In[5]:


index_future_months = pd.date_range(start='2024-01-01',end='2025-01-01', freq='M')
index_future_months = index_future_months.strftime('%b-%y')
print(index_future_months)
pred2 = model.predict(start = len(data),end = len(data)+11,type = 'levels').rename('ARIMA PRedictions')
pred2.index = index_future_months
print(pred2)

