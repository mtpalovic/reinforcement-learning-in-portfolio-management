#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

from numpy.random import randint

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import random

from datetime import datetime
import seaborn as sns

import pandas as pd
import math

import time as t

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (normalize, 
                                   StandardScaler, 
                                   MinMaxScaler, 
                                   LabelEncoder, 
                                   OneHotEncoder)

from sklearn.decomposition import PCA

from sklearn.feature_selection import (VarianceThreshold, 
                                       SelectKBest, 
                                       mutual_info_classif)

from sklearn.svm import SVC



from sklearn.metrics import (confusion_matrix,
                             classification_report,
                             accuracy_score,
                             precision_score)

from sklearn.utils import shuffle

from sklearn.model_selection import (train_test_split, 
                                     GridSearchCV, 
                                     cross_validate)

from sklearn.utils.estimator_checks import check_estimator

from typing import Union


# In[27]:


class s():
    
    TEST_SIZE = 0.3
    N_ITERS = 1e3
    TRAILS = 50
    
    
    def p_c(m):
        """Checks the type of init params (float, int, str).
        :param m: init method
        :return: init method after params are checked
        """
        
        #ref refers to self, arbitrary
        def f(ref, estimator, k, C, gamma, random_number):
            
            n = 0
            n = int(n)
            for p in [C, gamma]:
                try:
                    p + n
                except TypeError as e:
                    print(e)
                    
            
            j = "v"
            j = str(j)
            for v in [estimator, k]:
                try:
                    v + j
                except TypeError as e:
                    print(e)
                
            
            return m(ref, estimator, k, C, gamma, random_number)
        
        return f
    
    
    
    @p_c
    def __init__(self, 
                 estimator:str = "SVC", 
                 k:str = "linear", 
                 C:int = 1000, 
                 gamma = 1, 
                 random_number:int = None):
        
        
        
        #init param restriction on the dict
        self.kernels = {
            "linear": self.kernel_linear,
            "rbf": self.kernel_rbf,
            "poly": self.kernel_poly
        }
        
        if k not in self.kernels.keys():
            try:
                s = self.kernels[k]
            except KeyError as o:
                print(f"The kernel {k} is not in {self.kernels.keys()}")
        else:
            self.k = k
            
        
        #init param restriction on the size
        if 0.1 <= C <= 10000:
            self.C = C
        else:
            self.C = None

        
        
        self._estimator = estimator
        self._gamma = self.d_(gamma)
        self.random_number = random_number if random_number is not None else np.random.randint(0,100,size=None)
        
            
            
    
    
    
    
    @property
    def estimator(self):
        """Property decorator read only attribute. Instance attr cannot be changed. Getter method.
        :return: 
        :rtype: str
        """
        return self._estimator
    
    @estimator.setter
    def estimator(self,v):
        self._estimator = v
        
    
    
    
    @property
    def gamma(self):
        """init initiates instance attr and this method returns private attr
        :return: without this method private attr cannot be accessed (encapsulation)
        :rtype: 
        """
        return self._gamma
    
    @gamma.setter
    def gamma(self, h):
        if 0 <= h <= 1000:
            self._gamma = h
        else:
            self._gamma = None
        

            
             
            
    @staticmethod        
    def d_(val):
        """Check init param type before it is set in the init method. Only int type.
        :param val: val in the init method
        :type val: 
        """
        if not isinstance(val,int):
            raise TypeError("val must be of type int")
        return val
    
    
    
    
    
    def cA(self, a:str = None, l = None):
        """Check class attribute, not init (instance) attr. Other methods: getattr, setattr, hasattr.
        Init params are instance attributes, class attributes are defined outside constructor.
        :param u: defaults to None
        :type u: bool
        
        :raises AttributeError: if default not provided in getattr
        
        :return:
        :rtype:
        """
        
        if l is None:
            try:
                k = getattr(s,a)
            
            except AttributeError as e:
                
                print(e)
        
        else:
            pass
        
        return float(k)
        
        
    
    
    def __str__(self):
        """String representation of an object. Init params are instance attributes.
        :return: init param as string
        :rtype: str
        """
        return f"init params:{self.estimator},{self.k}, {self.C}, {self.gamma}, {self.random_number}"
    
    
    
    def create_tuple(self):
        """Creates a tuple
        :return:
        :rtype: tuple
        """
        self.tuple = ()
        
        return self.tuple
    
    
    
    def create_arr(self, *args):
        """Creates a list
        :return:
        :rtype: list
        """
        
        self.arr = [*args]
        
        return self.arr
    
    
    
    def create_dict(self):
        """Creates a dictionary
        :return:
        :rtype:
        """
        self.dict = {}
        
        return self.dict
    
    
    
    def load_data(self):
        """Creates data.
        :return: data
        :rtype: dataframe
        """
        path = "C:/Users/mpalovic/Desktop"
        ticker = "gspc"
        file_name = "ta.{}".format(str(ticker)) + ".csv"
        d_ = pd.read_csv(filepath_or_buffer = "{}/{}".format(path, file_name), 
                           parse_dates=["Date"], 
                           sep = ",")
        
        
        
        d = pd.DataFrame(d_)
        
        
        
        d.astype(
                {"Volume":float,"On Balance Volume":float}
            ).dtypes

        d.select_dtypes(np.number)
        
        
        d = d.iloc[:,2:3,:]
        
        
        
        
        return d
    
    
    
    
    def datetime_index(self):
        """Transforms dates into cos,sin
        :return: dataframe with dates adjusted
        :rtype: dataframe
        """
        pi = float(math.pi)
        
        d = self.load_data()
        mo = d["Date"].dt.month
        
        f = self.create_arr("cos", "sin")
    
        for i,v in enumerate(f):
            if i == 0:
                d[v] = np.cos(2 * pi * mo / mo.max())
            elif i == 1:
                d[v] = np.sin(2 * pi * mo / mo.max())
        
        d.drop(labels="Date", axis = 1, inplace = True)
        
        return d
    
    
    
    
    
    
    def mis_vals(self):
        """Check missing values
        :return: dataframe with missing vals checked
        :rtype: dataframe
        """
        d = self.datetime_index()
        
        for _ in d.select_dtypes(np.number):
            i = SimpleImputer(missing_values = np.nan, strategy = "mean")
            d = pd.DataFrame(i.fit_transform(d), 
                             columns = d.columns)
        
        return d
    
    
    
    
    def x_y_(self):
        
        d = self.mis_vals()
        
        pred = (d.shift(-7)["Close"] >= d["Close"]) #bool, if price in 14 days bigger, returns 1
        d.drop("Close", 1, inplace=True)
        pred = pred.iloc[:-7]
        d["Pred"] = pred.astype(int)
    
        d = d.dropna()
        x = pd.DataFrame(d.loc[:,d.columns != "Pred"])
        y = d.iloc[:,-1]
    
        return x,y
    
    
    
    
    
    
    def label_encoding(self):
        
        x,y = self.x_y_()
        
        if isinstance(y,np.ndarray):
            y = LabelEncoder().fit_transform(y)
        
        return y
    
    
    
    
    def kernel_linear(self, x, y):
        return np.dot(x,y.T)
    
    
    
    def kernel_rbf(self, x,y):
        
        #initialise k
        arr = np.zeros(x.shape[0], y.shape[0])
        
        for i,x in enumerate(x):
            for j,y in enumerate(y):
                arr[i,j] = np.exp(-1*np.linalg.norm(x-y)**2)
        return arr
    
    
    
    def kernel_poly(self,x,y,p=3):
        return (1+np.dot(x,y))**p
    
    
    
    
    
    
    
    def feature_selection(self, num: Union[float,int] = 1.0*10e-3):
        
        
        x, _ = self.x_y_()
        c = []    
    
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)
        
        if num:
            
            #actual variance of cols
            x.var()

            selector = VarianceThreshold(num)
            selector.fit_transform(x)
            selector.get_support()
            high_var_cols = [int(i) for i,e in enumerate(selector.get_support(indices = False))]


            x = pd.DataFrame(x)
            if isinstance(x, pd.DataFrame):
                c = [x for z,x in enumerate(x.columns) if z not in high_var_cols]

            x = np.array(x[x.columns[~x.columns.isin(c)]])
        
        
        
        else:
            pass
        
        return x if num else self.scale()
        
    
    
    
    
    
    
    def selectK(self, n:int=10):
        if n:
            x = self.feature_selection() 
            y = self.label_encoding()
        
        
            n = int(n)
    
            
            high_score_features = self.create_arr()
            
            
            
            feature_scores = mutual_info_classif(x, y, random_state=self.random_number)
            
            
            
            
            
            x = pd.DataFrame(x)
            
            if feature_scores.any():
                for score, col in sorted(zip(feature_scores, x.columns), reverse=True)[:n]:
                    #print(f_name, round(score, 4))
                    high_score_features.append(col)
                    x_ = x[high_score_features]
        
        else:
            pass
        
        
        
        return x_
    
    
    
    
    
    
    def get_params(self, deep = True):
        
        par = {
            "Kernel": self.k,
            "C": self.C,
            "Rand num": self.random_number
        } 
        
        return par
        
            
            
        
    def return_params(self):
        count = len(self.get_params().keys())
        count = int(count)
        
        params = [f"{param}:{value}" for param, value in sorted(self.get_params(deep=True).items())[:count]]
                
        return params
    
    
    
    
    def data_split(self):
        
        x_ = self.selectK()
        x = x_.copy()
        
        
        y = self.label_encoding()
        
        
        
    
        self.n_samples, self.n_features = x.shape
    
        
        x_train, x_test, y_train, y_test = train_test_split(x, 
                                                        y, 
                                                        test_size=s.TEST_SIZE, 
                                                        random_state=self.random_number, 
                                                        shuffle = True,
                                                        stratify=y
                                                    )
        
        return x_train, x_test, y_train, y_test
    
    
    
    def scale(self):
    
        x_train, x_test, _, _ = self.data_split() 
        scaler = StandardScaler()
        
        
        
        x_train = scaler.fit_transform(x_train.values.reshape(-1,self.n_features))
        x_test = scaler.transform(x_test.values.reshape(-1,self.n_features))
        
        
        return x_train, x_test
        
    
    
    
    
    def fit(self):
        
        #linear hyperparams
        linear_params = {
           "kernel": "linear",
            "C": 1e3
        }
        
        #poly hyperparams
        poly_params = {
            "kernel": "poly",
            "C": 1e3,
            "degree": np.random.randint(2,3) #low inclusive, high exclusive
        }
        
        #rbf hyperparams
        rbf_params = {
            "kernel": "rbf",
            "C": 1e3,
            "gamma": 0.1
        }
        
        
        
        
        #the below func must run before fit, without parenthesis
        if callable(self.scale):
        
            #import from here 
            x_train, x_test = self.scale()
        
        
        
            #pass an an np.ndarray
            x_train = np.array(x_train)
        
        
            if isinstance(x_train, np.ndarray):
            
                #do not import x_train, x_test because I have modified it
                _, _, y_train, y_test = self.data_split()
            
            
                #checks if kernel exists
                if self.k in self.kernels.keys():
                
                    estimator = SVC(kernel = self.kernels[self.k])
                    model = estimator.fit(x_train, y_train)
            
            
                    y_pred = model.predict(x_test)
                    self.accuracy = accuracy_score(y_pred, y_test)
            
            
                else:
                    print(f"{self.k} required to be in {self.kernel.keys()}")
                    
                    if self.k == "linear":
                        SVC(C = linear_params["C"], kernel = linear_params["kernel"])
                    
                    elif self.k == "poly":
                        SVC(C = poly_params["C"], kernel = poly_params["kernel"], degree = poly_params["degree"])
                        
                    else:
                        SVC(C = rbf_params["C"], kernel = rbf_params["kernel"], gamma=rbf_params["gamma"])
                    
                    
    
            else:
                pass
        
        else:
            pass
        
        
        
        return self.accuracy
    
    
    
    
    
    def accuracy_score(self):
        
        
        x_train, x_test = self.scale()
        _, _, y_train, y_test = self.data_split()
        
        
        if isinstance(x_train, np.ndarray):
            
            #creates dict
            classifiers = self.create_dict()
            
            for i in list(self.kernels.keys())[:len(self.kernels.keys())]:
                classifiers[i] = SVC(kernel=i)
        
        
        
        
            accr = self.create_dict()
        
            for algorithm, classifier in classifiers.items():
                classifier.fit(x_train, y_train)
                y_pred = classifier.predict(x_test)
                accr[algorithm] = accuracy_score(y_pred, y_test)
            
        
                
            for algorithm, accuracy in sorted(accr.items())[:len(accr.keys())]:
            
                #notice must be 2x %% in order to show % in print
                print("%s Accuracy: %.2f%%" % (algorithm, 100*accuracy))
            
        
        
        return accr


# In[28]:


if __name__ == "__main__":
    m = s(estimator="SVC", k = "linear", C = 1000, gamma = 1, random_number=None)
    m.__str__()
    m.create_arr()
    m.load_data()
    m.datetime_index()
    m.mis_vals()
    m.x_y_()
    m.label_encoding()
    m.data_split()
    m.scale()
    m.selectK()
    m.get_params()
    m.return_params()
    m.feature_selection()
    m.fit()
    m.accuracy_score()
    m.create_tuple()


# In[29]:


m.load_data()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




