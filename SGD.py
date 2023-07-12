#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np

def stochastic_gradient_descent(X, y, learning_rate=0.01, num_epochs=100, batch_size=1):
    num_samples, num_features = X.shape

    # Initialize model parameters
    theta = np.random.randn(num_features)

    for epoch in range(num_epochs):
        # Shuffle the training data
        permutation = np.random.permutation(num_samples)
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]

        for i in range(num_samples):
            xi = X_shuffled[i]
            yi = y_shuffled[i]
            
            
            error = xi.dot(theta) - yi
            gradient = xi * error
            #print(gradient)
            #print(error)
            
            theta -= learning_rate * gradient

    return theta


# In[11]:


np.random.seed(42)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.randn(100, 1)

theta = stochastic_gradient_descent(X, y, learning_rate=0.1, num_epochs=100)

# Print the learned parameters
print("Learned parameters:")
print("Intercept:", theta[0])
#print("Slope:", theta[1])


# In[6]:


gradient


# In[5]:


X.shape


# In[3]:


theta


# In[ ]:




