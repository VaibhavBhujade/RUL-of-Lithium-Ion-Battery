# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 11:51:23 2020

@author: Lenovo
"""

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import Dataset
dataset = pd.read_csv('training_data.csv')
X_train=dataset.iloc[:,0:1].values
y_train=dataset.iloc[:,5].values

test_data = pd.read_csv('testing_data.csv')
X_005=test_data.iloc[:,0:1].values
y_005=test_data.iloc[:,5].values

# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=True)

# Import keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
regressor = Sequential()

# Adding the input layer and first hidden layer
regressor.add(Dense(output_dim=2 ,init = 'uniform',activation = 'tanh',input_dim = 1))

# Adding the output layer
regressor.add(Dense(output_dim = 1,init = 'uniform',activation = 'linear'))

# Compiling the ANN
regressor.compile(optimizer = 'sgd', loss = 'mean_squared_error', metrics= ['mean_absolute_error','accuracy'])

# Fitting the ANN in the Training set
regressor.fit(X_train, y_train, batch_size = 1,nb_epoch = 2000)
# Predicting the Test set result
y_pred = regressor.predict(X_005)


t=[40]
y_pred = regressor.predict(t)

t[0]+=1
y_pred = regressor.predict(t)
print(t)


while(t[0]<165 and y_pred[0][0]>1.3):
    t[0]+=1
    y_pred = regressor.predict(t)
    
rul= t[0]  - 40  
weights = regressor.layers[0].get_weights()[0]
biases = regressor.layers[0].get_weights()[1]

from sklearn.metrics import r2_score
import sklearn
import math
R2= r2_score(y_005, y_pred)
mse = sklearn.metrics.mean_squared_error(y_005, y_pred)
rmse = math.sqrt(mse)


#Visualize
initial_capacity =  regressor.predict([0])
print(initial_capacity)
threshold = initial_capacity*0.7
#Visual results training
plt.plot(0,1.68,color='yellow')
plt.plot(X_005, y_005, color='red')
plt.plot(X_005, y_pred, color='blue')
plt.axhline(threshold, color='yellow', linestyle='--')
plt.title('RUL Prediction')
plt.xlabel('Cycle')
plt.ylabel('Capacity (Ah)')
plt.show()

