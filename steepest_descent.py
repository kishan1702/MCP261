#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np

def steepest_descent(f, g, initial_params, learning_rate, num_iterations):
    params = initial_params
    for i in range(num_iterations):
        grad = g(params)
        params = params - learning_rate * grad
    return params

# Examples
def f(x):
    f = x**2 + 2*x + 1
    return f

def g(x):
    return 2*x + 2

initial_params = 10.0
learning_rate = 0.1
num_iterations = 100

optimized_params = steepest_descent(f, g, initial_params, learning_rate, num_iterations)

print("Optimized parameters:", optimized_params)
print("Optimized function value:", f(optimized_params))


# In[9]:


f =x**2 + 2*x + 1
np.gradient(f)


# In[ ]:




