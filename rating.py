#Max Kiluk
#4/13/2016

#[[i,j,r]]
#i = user
#j = movie
#r = rating

import numpy as np
import time

#load data, get user list, get mean ratings
data = np.load('train.npy')
userList = data[:,0]
meanRatings = np.mean(data[:,2])

#subtract mean ratings from actual ratings
data[:,2] = data[:,2] - meanRatings

#?
for j in range numMovies
    a[j] = np.mean(data[data[:,1] == j,2])
    data[data[:,1] = j,2] -= a[j]
     
