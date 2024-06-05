# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 19:08:55 2023

@author: Afshin
"""

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns



# Load the dataset
data = pd.read_csv('cleanThy Opt.csv')

# Explore the dataset
print(data.head())
print(data.info())
print(data.describe())


# Time series plot
plt.figure(figsize=(12, 6))
#plt.plot(data['Date'], data[' Last Price'], label='Closing Price')
plt.plot(data['Last Price'], label='Last Price')
plt.title('Turkish Airlines Stock Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Last Price')
plt.legend()
plt.show()

# Assuming 'Date' is a datetime column
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]


from sklearn.linear_model import LinearRegression

# Features and target variable for training set
#X_train = train[['Last Price', 'Diff', 'Volume Norm']]
#X_train = train[['Last Price', 'Lowest Price', 'Diff', 'Volume Norm']]
#X_train = train[['Last Price', 'Lowest Price', 'Highest Price', 'Volume Norm']]
#X_train = train[['Last Price', 'Diff', 'Highest Price', 'Volume Norm']]
###X_train = train[['Last Price', 'Diff', 'Highest Price', 'Volume']]
X_train = train[['Open Price', 'Diff', 'Highest Price', 'Volume']]
#X_train = train[['Diff', 'Highest Price', 'Volume']]

###y_train = train['Open Price']
y_train = train['Last Price']

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Features and target variable for testing set
#X_test = test[['Last Price', 'Diff', 'Volume Norm']]
#X_test = test[['Last Price', 'Lowest Price', 'Diff', 'Volume Norm']]
#X_test = test[['Last Price', 'Lowest Price', 'Highest Price', 'Volume Norm']]
#X_test = test[['Last Price', 'Diff', 'Highest Price', 'Volume Norm']]
###X_test = test[['Last Price', 'Diff', 'Highest Price', 'Volume']]
X_test = test[['Open Price', 'Diff', 'Highest Price', 'Volume']]
#X_test = test[['Diff', 'Highest Price', 'Volume']]

###y_test = test['Open Price']
y_test = test['Last Price']

# Predictions
predictions = model.predict(X_test)

#X_Full = data[['Open Price', 'Diff', 'Highest Price', 'Volume']]   #Try the model with whole dataset
#predictions_Full = model.predict(X_Full)    #Try the model with whole dataset

# Evaluate performance
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')


# Prediction plot
plt.figure(figsize=(12, 6))
#index1 = list(range(0,len(predictions_Full)))   #Try the model with whole dataset
index1 = list(range(0,len(predictions)))
#plt.plot(data['Date'], data['Last Price'], label='Closing Price')

#plt.plot(index1, predictions_Full, label='Open Price_Model')   #Try the model with whole dataset
plt.plot(index1, predictions, label='Open Price_Model')
plt.plot(index1, y_test, label='Open Price_Real')
#plt.plot(index1, data[['Last Price']], label='Open Price_Real')

plt.title('Turkish Airlines Stock_Model')
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.legend()
plt.show()
