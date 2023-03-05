from django.db import models
import pandas as pd 
import numpy as np
import pickle 
#from alpha_vantage.timeseries import TimeSeries
import time
from nsetools import Nse
from io import BytesIO
nse = Nse()

ITC = nse.get_quote('ITC')
ITC

import nsepy as nse
from datetime import date

end=date.today()

ITC_data_set = nse.get_history('ITC',index=False,start=date(2022,7,30),end=date(2023,2,10))
ITC_data_set['Date'] = ITC_data_set.index

# ITC_data_set.tail()

import matplotlib.pyplot as plt

import datetime

def str_to_datetime(s):
    split =s.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year=year, month=month, day=day)

import pandas as pd
import numpy as np

ITC_data_set.sort_index()

#Visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('Close price history')
plt.plot(ITC_data_set['Close'])
plt.xlabel('Date',fontsize = 25)
plt.ylabel('Close price USD($), fontsize=18')
# plt.show()

#create a new data fram with only the close column
data = ITC_data_set.filter(['Close'])
#convert the datafram to a numpy array
dataset = data.values
#Get the number of rows to train the model on
import math
training_data_len = math.ceil(len(dataset)* .8)

#SCALE THE DATA
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#Create the training data set
#create the scaled training data set
train_data = scaled_data[0:training_data_len,:]
#split the data into x_train and y_train data sets
x_train = []
y_train= []

for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    # if i<= 61:
    #     print(x_train)
    #     print(y_train)

#convert the x_train and y_train to numpy arrays
import numpy as np

x_train = np.array(x_train)
y_train = np.array(y_train)


#reshape the data
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train.shape

model = pickle.load(open('market/itc.pkl','rb'))

test_data = scaled_data[training_data_len - 60:,:]
#create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60,len(test_data)):
  x_test.append(test_data[i-60:1,0])
#convert the data to a numpy array
x_test = np.array(x_test, dtype=object)
x_test.shape
#reshape the data
x_test = np.reshape(x_test,(x_test.shape[0],1))

# x_test.shape



#BUILD THE LSTM MODEL
predictions = model.predict(x_train)
predictions = scaler.inverse_transform(predictions)
#Get the root mean squared error
rmse = np.sqrt(np.mean(predictions-y_train)**2)
# rmse
# y_train
train = data[:training_data_len]
valid = data[training_data_len:]
plt.figure(figsize=(12,6))
plt.plot(y_test, "b", label = 'Original Price')
plt.plot(predictions, 'r', label =' Predicted price')
plt.xlabel("Date")
plt.ylabel('Price')
plt.legend()
plt.savefig('static/assets/img/ITC.png')
