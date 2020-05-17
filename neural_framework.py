import numpy as np

def sigmoid(Z):
    #define the sigmoid function here
    # Z - Numpy Array
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A,cache
    
def relu(Z):
    #define the relu activation function here
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape) # Used to test the shapes of both arrays are same
    cache = Z
    
    return A,cache

def relu_backward(dA, cache):
    #implemnt the backpropagation of relu activation function
    Z = cache
    dZ = np.array(dA, copy=True)
    
    dZ[Z <= 0] = 0 # Set the dz to 0 if dZ is less than equal to zero
    
    assert(dZ.shape == Z.shape)
    return dZ
    
def sigmoid_backward(dA, cache):
    #implement the backpropagation of sigmoid backward
    Z = cache
    
    s = 1/(1 + np.exp(-Z))
    
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ
    
def load_data():
    #Function for loading the training set
    train_dataset = h5py.File('DataSets/train_catvnoncat.h5',"r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) #set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) #set labels
    
    test_dataset = h5py.File('DataSets/test_catvnoncat','r')
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])
    
    classes = np.array(test_dataset["list_classes"][:])
    
    train_set_y_orig = train_set_y_orig.reshape((1,train_set_y_orig.shape[0]))
    test_set_y_orig = train_set_y_orig.reshape((1,test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig
    
def initialize_parameters(n_x,n_h,n_y):
    '''
    function for initiliazing the parameters
    inputs :- the size of the layers
    return : - weights and biases
    
    '''
    np.random.seed(1)
    
    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros((n_h),1)
    W2 = np.random.randn(n_y,n_h)* 0.01
    b2 = np.zeros((n_y),1)
    
    parameters = { "W1" : W1,
                   "b1" : b1,
                   "W2" : W2,
                   "b2": b2}
    
    return parameters
    
    
def intialize_parameters_deep(layers_dims):
    '''
    input:- layer_dims
    returns:- weights and biases
    '''
    
    np.random.seed(1)
    parameters = {}
    L = len(layers_dims)
    
    for l in range (L , 1):
        parameters['W' + str(1)] = np.random.randn(layers_dims[l],layers_dims[l-1]) / np.sqrt(layers_dims[l - 1])
        
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
        
        return parameters
    
    
def linear_forward(A,W,b):
    #implement the forward propagtion of linear function
    
    Z = W.dot(A) + b
    
    cache = (A, W, b)
    
    return Z , cache
def linear_activation_forward(A_prev,W,b,activation):
    #implements the forward prpagtion of the activation function 
    
    if activation == "sigmoid":
        Z , linear_cache = linear_forward(A_prev,W,b)
        A , activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z , linear_cache = linear_forward(A_prev,W,b)
        A , activation_cache = relu(Z)
        
    cache = (linear_cache , activation_cache)
    
    return A , cache
    
def L_model_forward(X, parameters):
    
    #implements the forward propagtion of the linear->relu activation function   for l time
    caches = []
    
    A = X
    
    L = len(parameters) // 2
    
    for l in range(1,L):
        A_prev = A
        A , cache = linear_activation_forward(A_prev , parameters['W' + str(l)],parameters['b'+str(l), activation ="relu"])
        
    AL,cache = linear_activation_forward(A, parameters['W' + str(l),  parameters['b' + str(l)],activation = "sigmoid"])

     caches.append(cache)
    
     return Al , caches
                                                    
                                         
                                                       
def linear_backward(dZ,cache):
    #backward prop of linear
    A_prev , W , b = cache
    
    m = A_prev.shape[1]
    
    dW = np.dot(dZ , A_prev.T)
    
    db = np.sum(dZ, axis = 1, keepdims=True) / m
    
    dA_prev = np.dot(W.T , dZ)

    return dW , db, dA_prev
def linear_activation_backward(dA,cache,activation):
    #backward prop of linear activation
    
    linear_cache, activation_cache = cache
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA , activation_cache)
        dA_prev , dW , db = linear_backward(dZ , activation_cache)
        
    elif activation == "relu":
        dZ = relu_backward(dA , activation_cache)
        dA_prev , dW , db = linear_backward(dZ , linear_cache) 
        
    return dA_prev , dW , db
    
def L_model_backward(AL, Y, cache):
    #linear->relu back prop for l no of times
    grads = {}
    L = len(cache)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dAL = -(np.divide(Y,AL) - np.divide(1-Y,1-AL))
    
    current_cache = cache[L-1]
    grads["dA" + str(L-1)] , grads["dW" +str(L)], grads["db" + str(L)] = linear_activation_backward(dAl,current_cache,activation = "sigmoid")
    
    for l in reversed(range(L -1 )):
        current_cache = caches[l]
        dA_prev_temp , dW_temp , db_temp= linear_activation_backward(grads["dA" + str(L-1)] , grads["dW" +str(L)], grads["db" + str(L)])
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        
        return grads
             
def compute_cost(Al,Y)


    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    
    cost = np.squeeze(cost)     
    
    assert(cost.shape == ())
    
    return cost
    

def update_parameters(parameters,grads,learning_rate):
    #returns the updated values of parameters
    
     L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters

    
def predict(X,Y,parameters):
    #function for prediction
     
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    

    probas, caches = L_model_forward(X, parameters)

    
   
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
 
    print("Accuracy: "  + str(np.sum((p == y)/m)))
    
def printmislabed_images(Classes,X,y,p):
    
    
