# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 17:59:14 2018

"""
## Mixture of Guassians
__author__ = 'Santanu '

import numpy as np
from scipy.stats import multivariate_normal

# Function to generate Mixture of Guassian datapoints
def generate_data(prior,param,num_dp,num_cluster):
    #data = np.array()
    i = 0
    cluster = np.random.choice(num_cluster,num_dp,prior)
    for c in cluster:
        mean_ = param[c]['mean']
        cov_  = param[c]['cov']
        sample = np.random.multivariate_normal(mean_,cov_,1).T
        if i == 0:
            data = sample
        else:
            data = np.hstack((data,sample))
        i += 1
    return np.array(data)


prior = [0.3,0.4,0.3]
param = {0:{'mean':[0,0],'cov':[[1, 0.5],[0.5,1]]},1:{'mean':[11,11],'cov':[[1, -0.3],[-0.3,1]]},2:{'mean':[40,40],'cov':[[1, 0.2],[0.2,1]]}}
num_dp = 10000
num_cluster = 3

data = generate_data(prior,param,num_dp,num_cluster)
np.shape(data)

import matplotlib.pyplot as plt
plt.scatter(data[0,:],data[1,:])

# Expectation Maximization for Mixture of Guassians
def e_m(data,num_cluster,num_iter):
    num_dimension,num_data_points = np.shape(data)[0],np.shape(data)[1]
    dict_ = {}
    for i in range(num_cluster):
        rand_mean = np.random.randint(num_data_points)
        dict_[i] = {}
        dict_[i]['mean'] = data[:,rand_mean].T
        dict_[i]['cov']  = np.identity(num_dimension)
        
    c = np.ones(3)/3.0
    for iter in range(num_iter):
        prob_ = []
        for j in range(num_cluster):
            print('cluster:',j)
            mean_ = np.reshape(dict_[j]['mean'],(1,num_dimension))
            prob =  c[j]*multivariate_normal.pdf(data.T,mean=mean_[0], cov=dict_[j]['cov'])
            prob_.append(prob)
        prob_ = np.array(prob_)    
        norm_sum = np.sum(prob_,axis=0)
        for j in range(num_cluster):
            prob = prob_[j,:]/norm_sum
            print(np.shape(dict_[j]['cov']))
            dict_[j]['mean'] = np.dot(data,prob)/np.sum(prob)
            mean_ = np.matrix(dict_[j]['mean']).T
            inter_ = np.multiply(prob,data -  mean_)
            #inter_ = np.reshape(inter_,(num_dimension,num_records))
            cov_ = np.matrix(inter_)*np.matrix(data -  mean_).T/np.sum(prob)
            dict_[j]['cov'] = np.array(cov_) 
            c[j] = np.sum(prob)
        c = c[:]/np.sum(c) 
        print(dict_,c)
    return dict_,c 

dic1,c1 = e_m(data,3,10)
print dic1
print c1
