#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from deepdow.utils import raw_to_Xy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import gym
import random
from collections import deque
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent

import yfinance as yf

import math
from gym import spaces, logger
from gym.utils import seeding
from gym.envs.registration import register


# In[ ]:


class env():
    def __init__(self,
                 data = data
                 window_length=50,
                 portfolio_value= 10000, 
                 trading_cost= 0.25/100,
                 interest_rate= 0.02/250, 
                 train_size = 0.7):
    
    
    self.portfolio_value = portfolio_value
    self.window_length=window_length
    self.trading_cost = trading_cost
    self.interest_rate = interest_rate
    
    
    
    
    self.data = data
    
    self.features = self.data.shape[0]
    self.instruments = self.data.shape[1]
    self.time_periods = self.data.shape[2]
    
    self.end_train = int((self.data.shape[2]-self.window_length)*train_size)
    
    
    
    
    
    self.index = None
    self.state = None
    self.done = False
    
    
    #init seed
    self.seed()
    
    
    
    
    
    def return_pf(self):
        return self.portfolio_value
    
    
    
    
    
    def readTensor(self,X,t):
        return X[:,:,t-self.window_length:t]
    
    
    
    def readUpdate(self, t):
        #return the return of each stock for the day t 
        return np.array([1+self.interest_rate]+self.data[-1,:,t].tolist())
    
    
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    
    
    def reset(self, w_init, p_init, t=0 ):
        
        """ 
        This function restarts the environment with given initial weights and given value of portfolio
        """
        self.state= (self.readTensor(self.data, self.window_length) , w_init , p_init )
        self.index = self.window_length + t
        self.done = False
        
        return self.state, self.done
    
    
    
    
    
    
    
    def step(self, action):
        """
        This function is the main part of the render. 
        At each step t, the trading agent gives as input the action he wants to do. So, he gives the new value of the weights of the portfolio. 
        The function computes the new value of the portfolio at the step (t+1), it returns also the reward associated with the action the agent took. 
        The reward is defined as the evolution of the the value of the portfolio in %. 
        """

        index = self.index
        #get Xt from data:
        data = self.readTensor(self.data, index)
        done = self.done
        
        #beginning of the day 
        state = self.state
        w_previous = state[1]
        pf_previous = state[2]
        
        #the update vector is the vector of the opening price of the day divided by the opening price of the previous day
        update_vector = self.readUpdate(index)

        #allocation choice 
        w_alloc = action
        pf_alloc = pf_previous
        
        #Compute transaction cost
        cost = pf_alloc * np.linalg.norm((w_alloc-w_previous),ord = 1)* self.trading_cost
        
        #convert weight vector into value vector 
        v_alloc = pf_alloc*w_alloc
        
        #pay transaction costs
        pf_trans = pf_alloc - cost
        v_trans = v_alloc - np.array([cost]+ [0]*self.nb_stocks)
        
        #####market prices evolution 
        #we go to the end of the day 
        
        #compute new value vector 
        v_evol = v_trans*update_vector

        
        #compute new portfolio value
        pf_evol = np.sum(v_evol)
        
        #compute weight vector 
        w_evol = v_evol/pf_evol
        
        
        #compute instanteanous reward
        reward = (pf_evol-pf_previous)/pf_previous
        
        #update index
        index = index+1
        
        #compute state
        
        state = (self.readTensor(self.data, index), w_evol, pf_evol)
        
        if index >= self.end_train:
            done = True
        
        self.state = state
        self.index = index
        self.done = done
        
        return state, reward, done


# In[ ]:





# In[ ]:


trading_period = data.shape[2]
nb_feature_map = data.shape[0]
nb_stocks = data.shape[1]


# In[ ]:


# fix parameters of the network
m = nb_stocks


# In[ ]:


#dictionaries of the problem
dict_hp_net = {"n_filter_1": 2, 
               "n_filter_2": 20, 
               "kernel1_size":(1, 3)
              }

dict_hp_pb = {"batch_size": 50, 
              "ratio_train": 0.6,
              "ratio_val": 0.2, 
              "length_tensor": 10,
              "ratio_greedy":0.8, 
              "ratio_regul": 0.1
             }

dict_hp_opt = {"regularization": 1e-8, 
               "learning": 9e-2
              }

dict_fin = {"trading_cost": 0.25/100, 
            "interest_rate": 0.02/250, 
            "cash_bias_init": 0.7
           }

dict_train = {"pf_init_train": 10000, 
              "w_init_train": "d", 
              "n_episodes":2, 
              "n_batches": 10
             }

dict_test = {"pf_init_test": 10000, 
             "w_init_test": 'd'
            }


# hyperparameters of the neural network
n_filter_1 = dict_hp_net["n_filter_1"]
n_filter_2 = dict_hp_net["n_filter_2"]
kernel1_size = dict_hp_net["kernel1_size"]




#hyperparameters of the overall problem

# Size of mini-batch during training
batch_size = dict_hp_pb["batch_size"]
# Total number of steps for pre-training in the training set
total_steps_train = int(dict_hp_pb["ratio_train"]*trading_period)

# Total number of steps for pre-training in the validation set
total_steps_val = int(dict_hp_pb["ratio_val"]*trading_period)

# Total number of steps for the test
total_steps_test = trading_period-total_steps_train-total_steps_val

# Number of the columns (number of the trading periods) in each input price matrix
n = dict_hp_pb["length_tensor"]

ratio_greedy = dict_hp_pb["ratio_greedy"]

ratio_regul = dict_hp_pb["ratio_regul"]






#hyperparameters of the optimisation

# The L2 regularization coefficient applied to network training
regularization = dict_hp_opt["regularization"]
# Parameter alpha (i.e. the step size) of the Adam optimization
learning = dict_hp_opt["learning"]

optimizer = tf.train.AdamOptimizer(learning)





#finance parameters
trading_cost= dict_fin["trading_cost"]
interest_rate= dict_fin["interest_rate"]
cash_bias_init = dict_fin["cash_bias_init"]






#pvm parameters
# Beta in the geometric distribution for online training sample batches
sample_bias = 5e-5







#training parameters
w_init_train = np.array(np.array([1]+[0]*m))

pf_init_train = dict_train["pf_init_train"]

n_episodes = dict_train["n_episodes"]
n_batches = dict_train["n_batches"]



#test parameters
w_init_test = np.array(np.array([1]+[0]*m))
pf_init_test = dict_test["pf_init_test"]


#other env params
w_eq = np.array(np.array([1/(m+1)]*(m+1)))
w_s = np.array(np.array([1]+[0.0]*m))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


class agent():
    
    
    def __init__(self, 
                 m, 
                 n, 
                 sess, 
                 optimizer,
                 trading_cost=trading_cost,
                 interest_rate=interest_rate,
                 n_filter_1=n_filter_1,
                 n_filter_2=n_filter_2):
    
        # parameters
        self.trading_cost = trading_cost
        self.interest_rate = interest_rate
        self.n_filter_1 = n_filter_1
        self.n_filter_2 = n_filter_2
        self.n = n
        self.m = m
        
        
        
        
        
        
        #input to the neural network, as a state, X_t and weight previous
        with tf.variable_scope("Inputs"):

            

            # The Price tensor
            #first dimension of a placeholder is a batch size, then shape of price tensor
            self.X_t = tf.placeholder(tf.float32, 
                                      shape = [None, nb_feature_map, self.m, self.n]
                                     ) 
            
            
            # weights at the previous time step
            # none refers to unknown batch size
            # m refers to num of assets, m = 10 + 1 (cash)
            self.W_previous = tf.placeholder(tf.float32, 
                                             shape = [None, self.m + 1]
                                            )
            
            
            
            
            # portfolio value at the previous time step
            self.pf_value_previous = tf.placeholder(tf.float32, [None, 1])
            
            
            # vector of Open(t+1)/Open(t)
            self.dailyReturn_t = tf.placeholder(tf.float32, [None, self.m])
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        #built a network
        with tf.variable_scope("Policy_Model"):
            
            
            # variable of the cash bias
            bias = tf.get_variable('cash_bias', shape=[
                                   1, 1, 1, 1], initializer=tf.constant_initializer(cash_bias_init))
            
            
            # shape of the tensor == batchsize
            shape_X_t = tf.shape(self.X_t)[0]
            
            # trick to get a "tensor size" for the cash bias
            self.cash_bias = tf.tile(bias, tf.stack([shape_X_t, 1, 1, 1]))
            
            
            #first layer within the neural network
            with tf.variable_scope("Conv1"):
                # first layer on the X_t tensor
                # return a tensor of depth 2
                self.conv1 = tf.layers.conv2d(
                    inputs=tf.transpose(self.X_t, perm=[0, 3, 2, 1]),
                    activation=tf.nn.relu,
                    filters=self.n_filter_1,
                    strides=(1, 1),
                    kernel_size=kernel1_size,
                    padding='same')
                
            
            
            with tf.variable_scope("Conv2"):
                
                #feature maps
                self.conv2 = tf.layers.conv2d(
                    inputs=self.conv1,
                    activation=tf.nn.relu,
                    filters=self.n_filter_2,
                    strides=(self.n, 1),
                    kernel_size=(1, self.n),
                    padding='same')
                
                
            with tf.variable_scope("Tensor3"):
                #w from last periods
                # trick to have good dimensions
                w_wo_c = self.W_previous[:, 1:]
                w_wo_c = tf.expand_dims(w_wo_c, 1)
                w_wo_c = tf.expand_dims(w_wo_c, -1)
                self.tensor3 = tf.concat([self.conv2, w_wo_c], axis=3)
            
            
            
            with tf.variable_scope("Conv3"):
                #last feature map WITHOUT cash bias
                self.conv3 = tf.layers.conv2d(
                    inputs=self.conv2,
                    activation=tf.nn.relu,
                    filters=1,
                    strides=(self.n_filter_2 + 1, 1),
                    kernel_size=(1, 1),
                    padding='same')
                
            with tf.variable_scope("Tensor4"):
                #last feature map WITH cash bias
                self.tensor4 = tf.concat([self.cash_bias, self.conv3], axis=2)
                # we squeeze to reduce and get the good dimension
                self.squeezed_tensor4 = tf.squeeze(self.tensor4, [1, 3])
                
            
            
            with tf.variable_scope("Policy_Output"):
                # softmax layer to obtain weights
                self.action = tf.nn.softmax(self.squeezed_tensor4)
                
                
                
                
                
            with tf.variable_scope("Reward"):
                # computation of the reward
                #please look at the chronological map to understand
                constant_return = tf.constant(
                    1+self.interest_rate, shape=[1, 1])
                cash_return = tf.tile(
                    constant_return, tf.stack([shape_X_t, 1]))
                y_t = tf.concat(
                    [cash_return, self.dailyReturn_t], axis=1)
                Vprime_t = self.action * self.pf_value_previous
                Vprevious = self.W_previous*self.pf_value_previous

                # this is just a trick to get the good shape for cost
                constant = tf.constant(1.0, shape=[1])

                cost = self.trading_cost *                     tf.norm(Vprime_t-Vprevious, ord=1, axis=1)*constant

                cost = tf.expand_dims(cost, 1)

                zero = tf.constant(
                    np.array([0.0]*m).reshape(1, m), shape=[1, m], dtype=tf.float32)

                vec_zero = tf.tile(zero, tf.stack([shape_X_t, 1]))
                vec_cost = tf.concat([cost, vec_zero], axis=1)

                Vsecond_t = Vprime_t - vec_cost

                V_t = tf.multiply(Vsecond_t, y_t)
                self.portfolioValue = tf.norm(V_t, ord=1)
                self.instantaneous_reward = (
                self.portfolioValue-self.pf_value_previous)/self.pf_value_previous
                
                
                
                
            with tf.variable_scope("Reward_Equiweighted"):
                constant_return = tf.constant(
                    1+self.interest_rate, shape=[1, 1])
                cash_return = tf.tile(
                    constant_return, tf.stack([shape_X_t, 1]))
                y_t = tf.concat(
                    [cash_return, self.dailyReturn_t], axis=1)
  

                V_eq = w_eq*self.pf_value_previous
                V_eq_second = tf.multiply(V_eq, y_t)
        
                self.portfolioValue_eq = tf.norm(V_eq_second, ord=1)
            
                self.instantaneous_reward_eq = (
                self.portfolioValue_eq-self.pf_value_previous)/self.pf_value_previous
                
                
                
                
            with tf.variable_scope("Max_weight"):
                self.max_weight = tf.reduce_max(self.action)
                print(self.max_weight.shape)
                
                
            
            
            
            with tf.variable_scope("Reward_adjusted"):
                
                self.adjested_reward = self.instantaneous_reward - self.instantaneous_reward_eq - ratio_regul*self.max_weight
                
                
                
                
        #objective function 
        #maximize reward over the batch 
        # min(-r) = max(r)
        self.train_op = optimizer.minimize(-self.adjested_reward)
        
        # some bookkeeping
        self.optimizer = optimizer
        self.sess = sess
        
        
        
    
    
    
    
    
    
    
    
    
    
    #TWO MAIN METHODS, the above is just the init method
    
    
    def compute_W(self, X_t_, W_previous_):
        """
        This function returns the action the agent takes 
        given the input tensor and the W_previous
        
        It is a vector of weight

        """

        return self.sess.run(tf.squeeze(
                                    self.action), 
                                     feed_dict={self.X_t: X_t_, 
                                                self.W_previous: W_previous_
                                                }
                            )
    
    
    
    
    
    
    
    
    def train(self, X_t_, W_previous_, pf_value_previous_, dailyReturn_t_):
        """
        This function trains the neural network
        maximizing the reward 
        the input is a batch of the differents values
        """
        self.sess.run(self.train_op, feed_dict={self.X_t: X_t_,
                                                self.W_previous: W_previous_,
                                                self.pf_value_previous: pf_value_previous_,
                                                self.dailyReturn_t: dailyReturn_t_
                                               }
                     )


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[28]:


x = np.array([[[56, 183, 1],
               [65, 164, 0],
                [21, 164, 0],
                  [65, 254, 0]],
              [[85, 176, 1],
               [44, 164, 0],
                  [85, 176, 1],
                   [44, 164, 0]]])


# In[29]:


#first num of dimensions, second row number, last col number
x.shape


# In[34]:


#index a tensor
x[1][1][1]


# In[20]:


x.shape[0] #dimensions are features


# In[21]:


x.shape[1] #row how many stocks


# In[22]:


x.shape[2] #col trading period


# In[ ]:


#selimamrouni
#https://github.com/selimamrouni/Deep-Portfolio-Management-Reinforcement-Learning/blob/master/DPM.ipynb
trading_period = data.shape[2]
nb_feature_map = data.shape[0]
nb_stocks = data.shape[1]


# In[ ]:


#jiang2017 switched rows and cols but features are the same


# In[43]:


x = tf.constant(np.arange(1, 13, dtype=np.int32),shape=[2, 2, 3])
x[1][0][2]


# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


#very good for basic understanding
#https://rubikscode.net/2021/07/13/deep-q-learning-with-python-and-tensorflow-2-0/


# In[ ]:


#The current state of the environment and the agent can be presented with the render method


# In[6]:


import pandas as pd


# In[9]:


# (i) large cap US equities
# (ii) middle cap US equities
# (iii) small cap US equities

# (iv) large and middle cap EU equities
# (v) small cap EU equities

# (vi) equities developing countries



# (vii) global government bonds
# (viii) emerging markets sovereingn bonds
# (ix) high yield bonds
# (x) corporate bonds






# equities

# (i) SXR8.DE iShares Core S&P 500 UCITS ETF USD (Acc)
# (ii) SPY4.DE SPDR S&P400 US Mid Cap UCITS ETF
# (iii) ZPRR.DE SPDR Russell 2000 US Small Cap UCITS ETF
# (iv) XSX6.DE DWS Stoxx Europe 600 UCITS ETF (DR) 1C, 
# (v) XXSC.DE DWS MSCI Europe Small Cap Index UCITS ETF (DR) 1C
# (vi) IS3N.DE Ishares Core MSCI EM IMI UCITS ETF


# fixed income

# (vii) DBZB.DE DWS Global Goverment Bond UCITS ETF (DR) 1C (EUR hedged)
# (viii) FRCK.DE UBS ETF (LU) Bloomberg Barclays USD Emerging Markets Sovereign UCITS ETF (hedged to EUR) A-acc
# (ix) XHYA.DE DWS EUR High Yield Corporate Bond UCITS ETF (DR) 1C
# (x) XBLC.DE DWS EUR Corporate Bond UCITS ETF (DR) 1C





symbols = ["SXR8.DE", "SPY4.DE", "ZPRR.DE", "XSX6.DE", "XXSC.DE", "IS3N.DE", "DBZB.DE", "FRCK.DE", "XHYA.DE", "XBLC.DE"]
start = "2019-01-01"
end = "2022-06-12"
    


sxr8 = yf.download("SXR8.DE","2019-01-01","2022-06-12")
spy4 = yf.download("SPY4.DE","2019-01-01","2022-06-12")
zprr = yf.download("ZPRR.DE","2019-01-01","2022-06-12")
xsx6 = yf.download("XSX6.L","2019-01-01","2022-06-12")
xxsc = yf.download("XXSC.L","2019-01-01","2022-06-12")
is3n = yf.download("IS3N.DE","2019-01-01","2022-06-12")


dbzb = yf.download("DBZB.DE","2019-01-01","2022-06-12")
frck = yf.download("FRCK.DE","2019-01-01","2022-06-12")
xhya = yf.download("XHYA.DE","2019-01-01","2022-06-12")
xblc = yf.download("XBLC.MI","2019-01-01","2022-06-12")



sxr8 = sxr8[:860]
spy4 = spy4[:860]
zprr = zprr[:860]
xsx6 = xsx6[:860]
xxsc = xxsc[:860]
is3n = is3n[:860]


dbzb = dbzb[:860]
frck = frck[:860]
xhya = xhya[:860]
xblc = xblc[:860]


data = np.stack([sxr8, spy4, zprr, xsx6, xxsc, is3n, dbzb, frck, xhya, xblc])
data.shape
a = np.moveaxis(data,[2,0,1],[0,1,2]).shape
a

#note a is of shape (f,n,m) features, time, assets, exactly as in jiang2017


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




