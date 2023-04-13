import numpy as np

#Using relu
def relu(val):
    return np.maximum(0, val)


#Derivative of relu, returns a copy of the OG array but with 1 or 0 depending on if the value is > 0
def relu_derivative(arr):
    return arr > 0

#The softmax function for the output layer
def softmax(z):
    z = z - np.max(z, axis = 1).reshape(z.shape[0], 1)
    return np.exp(z) / np.sum(np.exp(z), axis = 1).reshape(z.shape[0] , 1)
