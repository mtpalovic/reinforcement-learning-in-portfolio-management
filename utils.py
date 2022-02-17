#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


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




