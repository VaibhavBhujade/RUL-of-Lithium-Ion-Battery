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

annots5 = loadmat('C://Users//aakanksha//Desktop//ML BOOKS//B0005.mat')
annots6 = loadmat('C://Users//aakanksha//Desktop//ML BOOKS//B0018.mat')  
annots7 = loadmat('C://Users//aakanksha//Desktop//ML BOOKS//B0007.mat')  
annots18 = loadmat('C://Users//aakanksha//Desktop//ML BOOKS//B0018.mat')



signal_energy18 = list()
fluctuation_index18 = list()
skewness_index18 = list()
kurtosis_index18 = list()

for i in range(319):
    voltage18 = list()
    if annots6['B0018'][0][0][0][0][i][0]=='discharge':
       for j in range(0,179):
           voltage18.append(annots6['B0018'][0][0][0][0][i][3][0][0][0][0][j])
       mean6=statistics.mean(voltage18)
       sigma = 0
       for k in range(0,179):
           sigma=sigma+(voltage18[k]-mean6)**2;    
       w=1/18.5
       fluctuation_index18.append(np.sqrt(sigma)/w)
       stdev=statistics.stdev(voltage18)
       sigma2=0
       for k in range(0,179):
           sigma2=sigma2+(voltage18[k]-mean6)**3    
       skewness_index18.append(sigma2/stdev**3)
       sigma3=0
       for k in range(0,179):
           sigma3=sigma3+(voltage18[k]-mean6)**4    
       kurtosis_index18.append(sigma2/stdev**4)
       
       sum=0
       for j in range(1,179):
           sum=sum+(np.fabs(annots6['B0018'][0][0][0][0][i][3][0][0][0][0][j])**2 *
                    (annots6['B0018'][0][0][0][0][i][3][0][0][5][0][j]-annots6['B0018'][0][0][0][0][i][3][0][0][5][0][j-1]))
       signal_energy18.append(sum)
       
import csv

signal_energy006 = np.asarray(signal_energy18,dtype=float)
signal_energy006 = signal_energy006.reshape(len(signal_energy18),1)

data = {"Signal Energy":signal_energy18,"Fluctuation Index":fluctuation_index18,"Skewness Index":skewness_index18,"Kurtosis Index":kurtosis_index18}

df=pd.DataFrame(data)
df.to_csv("FeaturesB018.csv")