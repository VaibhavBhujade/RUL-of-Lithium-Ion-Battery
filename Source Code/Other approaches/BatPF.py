# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 12:04:25 2020

@author: user
"""

import math
from keras.optimizers import Optimizer
from keras import backend as K
import numpy as np
import random

if K.backend() == 'tensorflow':
    import tensorflow as tf


from scipy.stats import chi2 
from numpy.linalg import norm 

class BatPF(Optimizer):
    def __init__(self,D=7,NP=50,**kwargs):
        
        super(BatPF,self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.D=D
            self.NP=NP
            self.best=[0]*D
            self.x_obs=[0]*D
            self.z_obs=[0]*D
            self.x_pred=[[0 for i in range (D)] for j in range (NP)]
            self.z_pred=[[0 for i in range (D)] for j in range (NP)]
            self.v=[[0 for i in range (D)] for j in range (NP)]
            self.f=[0]*NP
            self.r=[0.5]*NP
            self.A=[0.5]*NP
            self.process_noise=np.random.normal(0,10,D)
            self.measurement_noise=np.random.normal(0,1,D)
            self.k=0
            self.Fitness=[0]*NP
            self.f_min=0.0
            self.fmin = 0.0
            self.fmax = 2.0
            
    
    def get_updates(self,loss,params):
        
        self.k+=1
        def state_transistion(x,k):
            return ((x/2)+(25*x/(1+x**2))+8*np.math.cos(1.2*k))    
    
        def measurement(x):
            return x*x/20
        #eq 6 and 7
        for j in range (self.D):
            self.x_obs[j]=state_transistion(self.x_obs[j],self.k)+self.process_noise[j]
            self.z_obs[j]=measurement(self.x_obs[j])+self.measurement_noise[j]
        #BAT algo 
        #objective function
        def objective(true,predicted,D):
            val=0.0
            for i in range(D):
                val= val+ math.exp(-0.5*(true[i]-predicted[i]))
            #print("val "+ str(val))
            return val
        #x=[x1,...,xd]
       
        #initialize bat population xi and vis
        #define fi at xi
        
        #Initialize ri and Ai
        x_new=[[0 for i in range (self.D)] for j in range (self.NP) ]
        t=0
        
        for i in range(self.NP):
            for j in range(self.D):
                rnd = np.random.uniform(0, 1)
                self.v[i][j] = 0.0
                self.x_pred[i][j] = rnd
                self.z_pred[i][j]=measurement(self.x_pred[i][j])+self.measurement_noise[j]
                #print("z_pred "+str(self.z_pred))
            self.Fitness[i] = objective(self.z_obs,self.z_pred[i],self.D)
            #print ("Fitness"+str(self.Fitness))
        
        i = 0
        j = 0
        for i in range(self.NP):
            #print(self.Fitness[i])
            #print(self.Fitness[j])
            if self.Fitness[i] < self.Fitness[j]:
                j = i
        for i in range(self.D):
            self.best[i] = self.x_pred[j][i]
        self.f_min = self.Fitness[j]
                
        #while(t<max no. iterations)
        while(t<1000):
        #{ Generate new solutions by adjusting fi,vi,xi using eq 12,13,14
            for i in range(self.NP):
                rnd = np.random.uniform(0, 1)
                self.f[i] = self.fmin + (self.fmax - self.fmin) * rnd
                for j in range(self.D):
                    self.v[i][j] = self.v[i][j] + (self.x_pred[i][j] -self.best[j]) * self.f[i]
                    self.x_pred[i][j] = self.x_pred[i][j] + self.v[i][j]
            
                rnd = np.random.random_sample()
                # if(random<ri)
                if(rnd<self.r[i]):
                    #{Select a solution among the best solutions
                    for j in range(self.D):
                        x_new[i][j]=self.best[j] + 0.001 * random.gauss(0, 1)
                    #Generate a local solution around the selected best solution
                    #xnew = xold + epsilon[-1,1]*Avg loudness
                    # }
                self.z_pred[i][j]=measurement(x_new[i][j])+self.measurement_noise[j]
                Fnew = objective(self.z_obs,self.z_pred[i],self.D)
        #Generate a solution by flying randomly
                rnd = np.random.random_sample()
        #if(random<Ai and I(xi)<I(x*))
                if rnd<self.A[i] and Fnew<self.Fitness[i]:
                    for j in range(self.D):
                        self.x_pred[i][j] = x_new[i][j]
                    self.Fitness[i] = Fnew
        #{Accept the new solutions
        #Increase ri (r=r0(1-exp(-gamma*t))) and reduce Ai(Ai(t+1)=alpha*Ai(t))}
        #Rank the bats and find the current best x*
        #}
                if Fnew <= self.f_min:
                    for j in range(self.D):
                        self.best[j] = x_new[i][j]
                    self.f_min = Fnew
            t+=1
        #End BAT algo
        
        
        #get x*
        #eq 10 (chi2pdf)
        i=0
        for p in params:
            new_p=p*K.constant([chi2.pdf(norm(self.z_obs[i]-self.best[i]),2)])
            self.updates.append(K.update(p,new_p))
            i+=1
            

        return self.updates
        #return self.updates
    def set_weights(self, weights):
        params = self.weights
        # Override set_weights for backward compatibility of Keras 2.2.4 optimizer
        # since it does not include iteration at head of the weight list. Set
        # iteration to 0.
        if len(params) == len(weights):
            self.weights = weights
        super(BatPF, self).set_weights(weights)   
    def get_config(self):
        print("cinfig me bhi aya")
        config = {'D': float(K.get_value(self.D)) }
        base_config = super(BatPF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
import pandas as pd

# Import Dataset
dataset = pd.read_csv('Cycle n Capacity.csv')
X=dataset.iloc[:,0:1].values
y=dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=True)

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
regressor.compile(optimizer = BatPF(), loss = 'mean_squared_error', metrics= ['mean_absolute_error','accuracy'])

# Fitting the ANN in the Training set
regressor.fit(X_train, y_train, batch_size = 1,nb_epoch = 1000)
# Predicting the Test set result
y_pred = regressor.predict(X_test)

weights = regressor.layers[0].get_weights()[0]
biases = regressor.layers[0].get_weights()[1]
weights1 = regressor.layers[1].get_weights()[0]
biases1 = regressor.layers[1].get_weights()[1]

numsum=0
densum=0
mean=0
summ=0
for i in range(0,128):
    summ=summ+y_test[i]
    
mean=summ/128

for i in range(0,128):
    numsum=numsum+(y_test[i]-y_pred[i])**2
    densum=densum+(y_test[i]-mean)**2
    
R2=1-(numsum/densum)
#import math
Rmse = math.sqrt(numsum/128)