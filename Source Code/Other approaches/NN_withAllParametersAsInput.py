

# Import Libraries
import pandas as pd

# Import Dataset
dataset = pd.read_csv('training_data.csv')
X_train=dataset.iloc[:,0:5].values
y_train=dataset.iloc[:,5].values

test_data = pd.read_csv('testing_data.csv')
X_005=test_data.iloc[:,0:5].values
y_005=test_data.iloc[:,5].values
# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=True)"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_005 = sc.transform(X_005)

# Import keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
regressor = Sequential()

# Adding the input layer and first hidden layer
regressor.add(Dense(output_dim=10 ,init = 'uniform',activation = 'tanh',input_dim = 5))

regressor.add(Dense(output_dim=5 ,init = 'uniform',activation = 'tanh'))

# Adding the output layer
regressor.add(Dense(output_dim = 1,init = 'uniform',activation = 'linear'))

# Compiling the ANN
regressor.compile(optimizer = 'sgd', loss = 'mean_squared_error', metrics= ['mean_absolute_error'])

# Fitting the ANN in the Training set
regressor.fit(X_train, y_train, batch_size = 1,nb_epoch = 2000)
# Predicting the Test set result
y_pred = regressor.predict(X_005)

import matplotlib.pyplot as plt
listt = X_005[0]
init_capacity = 0.7 * regressor.predict(listt)
x_inp = []
for i in range(1,len(y_pred)+1):
    x_inp.append(i)
plt.plot(X_train, y_train, color='voilet')
plt.plot(x_inp, y_005, color='red')
plt.plot(x_inp, y_pred, color='blue')
#plt.scatter(x=0, init_capacity)
#plt.axhline(init_capacity, color='yellow', linestyle='--')
plt.title('RUL Prediction')
plt.xlabel('Cycle')
plt.ylabel('Capacity (Ah)')
plt.show()
##############
