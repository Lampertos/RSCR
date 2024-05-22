#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def narma_next(values,rands,index, **conf):
    a = conf.get("coefficients", [0.3,0.05,1.5,0.1])
    n = conf.get("order", 10) # order of the NARMA series
    
    
    
    i = index
    y = values
    u = rands

    result = a[0] * y[i-1] + a[1] * y[i-1] * np.sum(y[i-n:n]) + a[2] * u[i-n] * u[i] + a[3]
    
    if n == 20:
        result = np.tanh(result) + 0.2

    # Compute next value
    return result


# In[3]:


def narma(length, **conf):
    
    order = conf.get("order", 10) # order of the NARMA series
    coefficients = conf.get("coefficients", [0.3,0.05,1.5,0.1])

        
    rands = np.random.uniform(0,0.5,length+1) # follow the bounds of the paper
    
#     values = np.concatenate((inits, np.zeros(length)))
    values =  np.zeros(length)
    # Sample step-wise
    end = values.shape[0]
    for t in range(end):
        values[t] = narma_next(values, rands, t, order = order, coefficients = coefficients) 
        
    return values

