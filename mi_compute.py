import pandas as pd
import numpy as np
from feature_selection_methods import mutual_infoKSG as mi

X1 = pd.read_csv("datasets\X1_t1.csv").transpose()
dataset = X1.values

#first, the mutual information between in and out
k_list=np.arange(5,25)

miMax=np.zeros((9,len(k_list)))
for index,k in enumerate(k_list):
    for j in range(9):
        miMax[j,index] = mi(dataset[j,:],dataset[-1,:],k)

minIndex = np.argsort(miMax,axis=0)+1 #index of the min
print(minIndex)
df = pd.DataFrame(miMax)

#then mutual information between the features

miMax2=np.zeros((8,8,len(k_list)))
for i in range(8):
    for j in range(8):
        for index,k in enumerate(k_list):
            miMax2[i,j,index] = mi(dataset[i,:],dataset[j,:],k)
            miMax2[i,i,index]=-float('inf')
minIndex2 = np.argmax(miMax2,axis=0)+1 #2D array
print(minIndex2)
