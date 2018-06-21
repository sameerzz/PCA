# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:18:00 2018

@author: Sxdubey
"""

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d


mu_vec1 = np.array([0,0,0])


class1_sample=np.random.rand(3,20)
class2_sample=np.random.rand(3,20)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 10   
ax.plot(class1_sample[0,:], class1_sample[1,:], class1_sample[2,:], 'o', markersize=8, color='blue', alpha=0.5, label='class1')
ax.plot(class2_sample[0,:], class2_sample[1,:], class2_sample[2,:], '^', markersize=8, alpha=0.5, color='red', label='class2')

all_sample=np.concatenate((class1_sample,class2_sample),axis=1)
mean_x=np.mean(all_sample[0,:])
mean_y=np.mean(all_sample[1,:])
mean_z=np.mean(all_sample[2,:])
mean_vector=np.array([[mean_x],[mean_y],[mean_z]])
# Finding the covariance matrix
cov_mat=np.cov([all_sample[0,:],all_sample[1,:],all_sample[2,:]])
eig_val, eig_vec=np.linalg.eig(cov_mat)


#Computing scatter matrix
scatter_matrix=np.zeros((3,3))
for i in range(all_sample.shape[1]):
    scatter_matrix=scatter_matrix+(all_sample[:,i].reshape(3,1)-mean_vector).dot((all_sample[:,i].reshape(3,1)-mean_vector).T)

#Find Eigen values and eigen vectors of Scatter Matrix
eig_val_scatter,eig_vec_scatter=np.linalg.eig(scatter_matrix)

# Print eigen and corresponding eigen values
for i in range(len(eig_val)):
    print("Eigen Vector {} \n{}".format(i+1,eig_vec[:,i].reshape(1,3).T))


#Check if Eigen vector follows the calculation
np.testing.assert_array_almost_equal(cov_mat.dot(eig_vec[:,0].reshape(1,3).T),eig_val[0]*(eig_vec[:,0].reshape(1,3).T))

#Choose the K eigen vectors with the largest eigen values
eig_pairs=[(np.abs(eig_val[i]),eig_vec[:,i]) for i in range(len(eig_val))]
eig_pairs.sort(key=lambda x:x[0],reverse=True)

#Load the data to the eigen basis
mat=np.hstack((eig_pairs[0][1].reshape(3,1),eig_pairs[1][1].reshape(3,1)))
transformed=mat.T.dot(all_sample)




#Plot the transformed data
plt.plot(transformed[0,0:20], transformed[1,0:20], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
plt.plot(transformed[0,20:40], transformed[1,20:40], '^', markersize=7, color='red', alpha=0.5, label='class2')








