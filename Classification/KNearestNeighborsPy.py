from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from collections import Counter

dataset= { 'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features= [5,7]

knn= []


for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0],ii[1], s= 100)

plt.show()
    

def euclidean_distance(plot1, plot2):
 return sqrt(((plot1[0]-plot2[0])**2) + ((plot1[1]-plot2[1])**2))

target_pt= [0,0]




def classifier(dataset, target_pt):
    
    group_1= 0
    group_2= 0
    for i in dataset:
        for ii in dataset[i]:
            knn.append([euclidean_distance(ii,target_pt),i])
    for i in sorted(knn[:3]):
        if(i[1]=='k'):
            group_1 += 1
        else:
            group_2 += 2
    
    if(group_1>group_2):
        gr= 'k'
    else:
        gr= 'r'
    return (gr)


print(classifier(dataset, [0,0]))













