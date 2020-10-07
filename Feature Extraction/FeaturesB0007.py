# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 18:31:52 2020

@author: user
"""

# -- coding: utf-8 --
"""
Created on Sat Feb 15 21:44:50 2020

@author: disha
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import pandas as pd
import statistics

annots5 = loadmat('C://Users//Lenovo//Desktop//Machine Learning//My Projects//Battery//B0005.mat')
annots6 = loadmat('C://Users//Lenovo//Desktop//Machine Learning//My Projects//Battery//B0006.mat')  
annots7 = loadmat('C://Users//Lenovo//Desktop//Machine Learning//My Projects//Battery//B0007.mat')  
annots18 = loadmat('C://Users//Lenovo//Desktop//Machine Learning//My Projects//Battery//B0018.mat')

signal_energy5 = list()
fluctuation_index5 = list()
skewness_index5 = list()
kurtosis_index5 = list()

for i in range(616):
    voltage5 = list()
    if annots5['B0005'][0][0][0][0][i][0]=='discharge':
       for j in range(0,179):
           voltage5.append(annots5['B0005'][0][0][0][0][i][3][0][0][0][0][j])
       mean5=statistics.mean(voltage5)
       sigma = 0
       for k in range(0,179):
           sigma=sigma+(voltage5[k]-mean5)**2;    
       w=1/18.5
       fluctuation_index5.append(np.sqrt(sigma)/w)
       stdev=statistics.stdev(voltage5)
       sigma2=0
       for k in range(0,179):
           sigma2=sigma2+(voltage5[k]-mean5)**3    
       skewness_index5.append(sigma2/stdev**3)
       sigma3=0
       for k in range(0,179):
           sigma3=sigma3+(voltage5[k]-mean5)**4    
       kurtosis_index5.append(sigma2/stdev**4)
       
       sum=0
       for j in range(1,179):
           sum=sum+(np.fabs(annots5['B0005'][0][0][0][0][i][3][0][0][0][0][j])**2 *
                    (annots5['B0005'][0][0][0][0][i][3][0][0][5][0][j]-annots5['B0005'][0][0][0][0][i][3][0][0][5][0][j-1]))
       signal_energy5.append(sum)
       
import csv

signal_energy005 = np.asarray(signal_energy5,dtype=float)
signal_energy005 = signal_energy005.reshape(len(signal_energy5),1)

data = {"Signal Energy":signal_energy5,"Fluctuation Index":fluctuation_index5,"Skewness Index":skewness_index5,"Kurtosis Index":kurtosis_index5}

df=pd.DataFrame(data)
df.to_csv("FeaturesB005.csv")