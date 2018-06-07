# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 20:23:37 2018

@author: santanu
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import sys

class recommender:
    
    def __init__(self,num_users,num_items):
        self.ranks = 5
        self.batch_size = 32
        self.epochs = 500
        self.learning_rate = 1e-4
        self.inputs = 1024
        self.num_hidden = 500
        self.num_movies = num_movies
        self.num_ranks = 5
        
        
    def __network(self):
        
        self.x  = tf.placeholder(tf.float32, [None,self.num_movies,self.num_ranks], name="x") 
        self.xr = tf.reshape(self.x, [-1,self.num_movies*self.num_ranks], name="xr") 
        self.W  = tf.Variable(tf.random_normal([self.num_movies*self.num_ranks,self.num_hidden], 0.01), name="W") 
        self.b_h = tf.Variable(tf.zeros([1,self.num_hidden],  tf.float32, name="b_h")) 
        self.b_v = tf.Variable(tf.zeros([1,self.num_movies*self.num_ranks],tf.float32, name="b_v")) 

## Converts the probability into discrete binary states i.e. 0 and 1 
        def sample_hidden(probs):
            return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))

        def sample_visible(logits):
            logits = tf.reshape(logits,[None,self.num_movies,self.num_ranks])
            return tf.multinomial(logits,1)

          
  
## Gibbs sampling step
        def gibbs_step(x_k):
            x_k = tf.reshape(x_k,[-1,self.num_movies*self.num_ranks]) 
            h_k = sample_hidden(tf.sigmoid(tf.matmul(x_k,W) + b_h)) 
            x_k = sample_visible(tf.add(tf.matmul(h_k,tf.transpose(W)),b_v))
            return x_k
## Run multiple gives Sampling step starting from an initital point     
        def gibbs_sample(k,x_k):
            for i in range(k):
                x_k = gibbs_step(x_k) 
# Returns the gibbs sample after k iterations
            return x_k

# Constrastive Divergence algorithm
# 1. Through Gibbs sampling locate a new visible state x_sample based on the current visible state x    
# 2. Based on the new x sample a new h as h_sample    
        self.x_s = gibbs_sample(self.k,self.x) 
        self.h_s = sample_hidden(tf.sigmoid(tf.matmul(self.x_s,self.W) + self.b_h)) 

# Sample hidden states based given visible states
        self.h = sample_hidden(tf.sigmoid(tf.matmul(self.x,self.W) + self.b_h)) 
# Sample visible states based given hidden states
        self.x_ = sample_visible(tf.matmul(self.h,tf.transpose(self.W)) + self.b_v))

# The weight updated based on gradient descent 
        self.size_batch = tf.cast(tf.shape(x)[0], tf.float32)
        self.W_add  = tf.multiply(self.lr/self.size_batch,tf.subtract(tf.matmul(tf.transpose(self.x),self.h),tf.matmul(tf.transpose(self.x_s),self.h_s)))
        self.bv_add = tf.multiply(self.lr/self.size_batch, tf.reduce_sum(tf.subtract(self.x,self.x_s), 0, True))
        self.bh_add = tf.multiply(self.lr/self.size_batch, tf.reduce_sum(tf.subtract(self.h,self.h_s), 0, True))
        self.updt = [self.W.assign_add(self.W_add), self.b_v.assign_add(self.bv_add), self.b_h.assign_add(self.bh_add)]
        
        
    def __train__(self):

# TensorFlow graph execution

        with tf.Session() as sess:
            # Initialize the variables of the Model
            init = tf.global_variables_initializer()
            sess.run(init)
            
            
            # Start the training 
            for epoch in range(self.num_epochs):
                if epoch < 150:
                    self.k = 2
    
                if (epoch > 150) & (epoch < 250):
                    self.k = 3
                    
                if (epoch > 250) & (epoch < 350):
                    self.k = 5
    
                if (epoch > 350) & (epoch < 500):
                    self.k = 9
                
                    # Loop over all batches
                for i in range(total_batch):
                    self.X_train = self.next_batch(self.input)
                    # Run the weight update 
                    #batch_xs = (batch_xs > 0)*1
                    _ = sess.run([self.updt],feed_dict={self.x:self.X_train})
                    
                # Display the running step 
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1))
                          
            print("RBM training Completed !")
            




        
                        
            
            
            
        
            
            
        
        
        
        
        
        
        
        
        
