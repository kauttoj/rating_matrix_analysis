# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 10:48:45 2017

@author: JanneK
"""

import pandas as pd
from surprise import SVD                                                       
from surprise import Dataset                                                   
from surprise import Reader                                                    
import numpy as np

data_full = np.array([[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
       [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
       [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
       [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
       [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
       [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
       [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]])

data = np.array([[  5.,   5.,   5.,  -1,   5.,  -1,  -1,  -1,  -1,   5.,   5.,  5.,   5.,   5.,   5.],
       [  7.,   7.,   7.,   7.,  -1,   7.,   7.,   7.,   7.,   7.,   7.,   7.,   7.,   7.,   7.],
       [  2.,   2.,   2.,  -1,   2.,   2.,   2.,   2.,   2.,   2.,   2.,   2.,  -1,   2.,  -1],
       [ -1,   3.,   3.,   3.,  -1,   3.,   3.,   3.,   3.,   3.,   3.,    3.,   3.,   3.,   3.],
       [ -1,  -1,  -1,   6.,   6.,   6.,  -1,   6.,   6.,   6.,   6.,     -1,  -1,  -1,   6.],
       [  9.,   9.,  -1,   9.,   9.,   9.,   9.,  -1,   9.,  -1,  -1,     -1,   9.,   9.,   9.],
       [  2.,  -1,   2.,   2.,   2.,  -1,   2.,   2.,  -1,  -1,  -1,       2.,   2.,  -1,  -1]])

users = []
items = []
file = open('mydata.csv','w')
ratings = []
siz = data.shape
for col in range(0,siz[1]):
    ind = np.where(data[:,col]>0)[0]
    for j in range(0,len(ind)):
#        users.append('user%i' % (col+1))
#        items.append('item%i' % (ind[j]+1))
        users.append(col+1)
        items.append(ind[j]+1)        
        ratings.append(data[ind[j],col])
        file.write('%s %s %s\n' % (users[-1],items[-1],ratings[-1]))

file.close()
                              
#ratings_dict = {'item': items,'rating': ratings,'user': users}
#df = pd.DataFrame(ratings_dict)
#reader = Reader(rating_scale=(1,10))
#obj = Dataset.load_from_df(df[['item','rating','user']], reader)

# As we're loading a custom dataset, we need to define a reader. In the
# movielens-100k dataset, each line has the following format:
# 'user item rating timestamp', separated by '\t' characters.
reader = Reader(line_format='user item rating',sep=' ',rating_scale=(1,10))
dataobj = Dataset.load_from_file('D:/GoogleDrive/mydata.csv', reader=reader)

traindata = dataobj.build_full_trainset()
                                               
algo = SVD(
    verbose =True,
    n_factors = 5,
    n_epochs  = 100)                                                               
                                                                
algo.train(traindata)

data_fill=data.copy()
for col in range(0,siz[1]):
    for row in range(0,siz[0]):
        #data_fill[row,col]=algo.predict('user%i' % (col+1),'item%i' % (row+1)).est
        data_fill[row,col]=algo.predict((col+1),(row+1)).est

print((np.round(data_fill)).astype(np.int))

print(data_full)
