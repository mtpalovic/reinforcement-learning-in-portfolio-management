#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#index np.array[0] means row
#index np.array[0,0] means row and col


# In[ ]:


#alpha - learning rate, should decrease as i learn
#gamma - if 1 then prefer long term revard, if 0 then prefer immediate revard
#gamma should decrease across time

#epsilon - exploration/exploitation, should decrease 
#if epsilon 1 then exploration and do not learn enough from policy
#if epsilon 0 then exploitation and learn much from policy


# In[8]:


import numpy as np
from numpy import linalg


# In[ ]:


a = np.array([[12,7,3],[4,5,6],[7,8,9]])
b = np.array([[5,8,1,2],[6,7,3,0],[4,5,9,1]])


# In[ ]:


r = np.zeros((3,4),dtype=int)


# In[ ]:


a


# In[ ]:


b


# In[ ]:


for i in range(len(a),1):
    
    for j in range(len(b[0]),1):
    
        for k in range(len(b),1):
            
            r[i][j] += a[i][k]*b[k][j]


# In[ ]:


i = np.array([1,2,3,4,5])
j = np.array([8.40,4.50,5.00,4.70,9.70])
q = np.dot(i,j)
q


# In[ ]:


a = 2 * [20]
a


# In[ ]:


# ak sme na epoch200 tak sprav check
#if epoch % sample_interval == 0:
#print(f"Episode: {epoch}") we are currently on epoch 
np.max
np.argmax
np.linalg
np.zeros()
#epoch,batch

#batch,epoch are hyperparams
#feature vector is a row vector
# sample - single row of data, also called input vector or feature vector
#batch - The batch size is a hyperparameter that defines the number of samples to work through before updating the internal model parameters
#epoch - how many times run through all dataset

#model updates after every batch

#200 samples, batch size of 5, 40 batches, then 1 epoch has 40 batches/200 samples


# In[ ]:


#np.max returns the maximum value from np.array
#np.argmax returns the INDEX of the maximum value from np.array


# In[5]:


#tuple of int
a = np.zeros(shape = (5,5), dtype = np.float64)
a


# In[ ]:


#if random.uniform(0, 1) < epsilon:
#    action = env.action_space.sample() # Explore action space
#else:
#    action = np.argmax(q_table[state]) # Exploit learned values
    
    
#if less than epsilon, take random action and continue exploring
#if greater than epsilon, index table by the state and choose the highest action value np.argmax

