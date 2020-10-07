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
annots6 = loadmat('C://Users//aakanksha//Desktop//ML BOOKS//B0006.mat')  
annots7 = loadmat('C://Users//aakanksha//Desktop//ML BOOKS//B0007.mat')  
annots18 = loadmat('C://Users//aakanksha//Desktop//ML BOOKS//B0018.mat')



signal_energy6 = list()
fluctuation_index6 = list()
skewness_index6 = list()
kurtosis_index6 = list()

for i in range(616):
    voltage6 = list()
    if annots6['B0006'][0][0][0][0][i][0]=='discharge':
       for j in range(0,179):
           voltage6.append(annots6['B0006'][0][0][0][0][i][3][0][0][0][0][j])
       mean6=statistics.mean(voltage6)
       sigma = 0
       for k in range(0,179):
           sigma=sigma+(voltage6[k]-mean6)**2;    
       w=1/18.5
       fluctuation_index6.append(np.sqrt(sigma)/w)
       stdev=statistics.stdev(voltage6)
       sigma2=0
       for k in range(0,179):
           sigma2=sigma2+(voltage6[k]-mean6)**3    
       skewness_index6.append(sigma2/stdev**3)
       sigma3=0
       for k in range(0,179):
           sigma3=sigma3+(voltage6[k]-mean6)**4    
       kurtosis_index6.append(sigma2/stdev**4)
       
       sum=0
       for j in range(1,179):
           sum=sum+(np.fabs(annots6['B0006'][0][0][0][0][i][3][0][0][0][0][j])**2 *
                    (annots6['B0006'][0][0][0][0][i][3][0][0][5][0][j]-annots6['B0006'][0][0][0][0][i][3][0][0][5][0][j-1]))
       signal_energy6.append(sum)
       
import csv

signal_energy006 = np.asarray(signal_energy6,dtype=float)
signal_energy006 = signal_energy006.reshape(len(signal_energy6),1)

data = {"Signal Energy":signal_energy6,"Fluctuation Index":fluctuation_index6,"Skewness Index":skewness_index6,"Kurtosis Index":kurtosis_index6}

df=pd.DataFrame(data)
df.to_csv("FeaturesB0006.csv")