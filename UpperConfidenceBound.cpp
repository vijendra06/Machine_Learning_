import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv('Ads_CTR_Optimisation.csv')
#random selection
import random
N= 10000
n=10
reward=0 
for i in range(0, N):
    ad= random.randrange(n)
    reward= reward+ dataset.values[i,ad]
#ucb
import math
N=10000
n=10
reward=0
noofselections=[0]*n
sumofrewards= [0]*n

for i in range(0,N):
    ad=0
    maxupperbound=0
    for j in range(0,10):
        if noofselections[j]> 0 :
            avgreward= sumofrewards[j] /noofselections[j]
            deltai= math.sqrt(3/2 * math.log(i+1) / noofselections[j])
            upperbound= avgreward + deltai
            
        else:
            upperbound=1000000
            
        if upperbound > maxupperbound:
            maxupperbound= upperbound
            ad=j
            
    noofselections[ad]+= 1
    sumofrewards[ad] += dataset.values[i, ad]
    reward += dataset.values[i, ad]
        