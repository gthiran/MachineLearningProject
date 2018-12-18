import numpy as np
import vq_methods as vqm
import tensorflow as tf
from tensorflow import keras
from keras import Sequential 
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
import pandas as pd



class model:
    def __init__(self):
        self.target = None
        self.y = None
        
    def meta_find(self,LM,validation_fun):
        pass
    
    def error(self):
        return np.sqrt(np.mean((self.y-self.target)**2)) 
    
    def _bias(X):
        return np.concatenate((np.ones((1,X.shape[1])),X),axis=0)
    
class linear_regression(model):
    def __init__(self,D):
        model()
        self.w = np.zeros((D,))
    
    def train(self,L):
        features = linear_regression._bias(L[:-1,:]) # features DxN
        target = L[-1,:]
        self.w = np.linalg.lstsq(features.T,target,rcond=None)[0] 
        # solve by minimizing norm2
    
    def evaluate(self,T):
        X = linear_regression._bias(T[:-1,:])
        self.target = T[-1,:]
        self.y = (self.w).T@X
        return self.y
    
class knn(model):
    def __init__(self):
        model()
        self.k = 4
        self.learning_data = None
        self.learning_target = None
        self.k_list= np.arange(1,51)
        
    def train(self,L):
        self.learning_data = L[:-1,:]
        self.learning_target = L[-1,:]
        
    def evaluate(self,T):
        y = np.empty(T[-1,:].shape)
        dist_array = np.empty((self.learning_data.shape[1],))
        
        for i in range(0,T.shape[1]):
            for j in range(0,self.learning_data.shape[1]):
                dist_array[j] = np.linalg.norm(T[:-1,i]-self.learning_data[:,j],
                                               ord=2,axis=0)
            
            min_index = np.argsort(dist_array)
            y[i] = np.mean(self.learning_target[min_index[:self.k+1]])
        
        self.target = T[-1,:]
        self.y = y
        return self.y
    
    def meta_find(self,LM,validation_fun):
        error_array = np.zeros(self.k_list.shape)
        for index,k in enumerate(self.k_list):
            self.k = k
            error_array[index] = validation_fun(self,LM)
            
        # best k
        self.k = self.k_list[np.argmin(error_array)]
        return (self.k,error_array)
    
class rbfn(model):
    #This code has largely been inspired by a the code written by Thomas Rückstieß
    #He can be found at the following address :-
    #http://www.rueckstiess.net/research/snippe-ts/show/72d2363e (18-12-11, 17:30)
    def __init__(self, D):
        #metaparameters
        self.h = 50                              #smoothing factor
        self.numCenters = 50
        self.alpha = 0.5
        self.beta = 1
        self.epochs = 10
        self.h_list= np.linspace(1,40,41)
        self.numCenters_list=np.arange(10,105,5)
        self.beta_list = np.array([1.5])
        maxi = np.max(self.numCenters_list)
        self.centers = np.zeros((D,maxi)) #c_i
        self.widths = np.zeros((maxi,))   #sigma_i
        self.W = np.zeros((maxi,))        #w_i
     
    def _calcAct(self, X):
        # calculate activations of RBFs
        N=X.shape[1]
        G = np.zeros((self.numCenters,N))
        for indexC in range(self.numCenters):
            s = np.linalg.norm((np.vstack(self.centers[:,indexC])-X),axis=0)
            if self.widths[indexC] == 0:
                G[indexC,:]=0
            else:
                G[indexC,:]= np.exp(-0.5*(s/self.widths[indexC])**2)
        return rbfn._bias(G)
     
    def train(self,L):
        N=L.shape[1]
        features = L[:-1,:] # features DxN
        target = L[-1,:] #1xN
        
        # choose center vectors from training set with fsl method
        self.centers = vqm.fsens_learning(features,self.alpha,self.beta,
                                          self.numCenters,self.epochs)
        
        # computes widths
        Number = np.zeros((self.numCenters,))
        DistCentroids = np.zeros((self.numCenters,))
        for i in range(N): #for each point, determine its centroïd
            Dist = np.linalg.norm((self.centers-np.vstack(features[:,i])),axis=0)
            
            indexMin=np.argmin(Dist)
            #and then add the distance
            DistCentroids[indexMin]+= Dist[indexMin]      
            Number[indexMin]+=1
           
        for i in range(self.numCenters):
            if Number[i] == 0:
                self.widths[i] = 0
            else:
                #mean of the distances for all the centroids
                self.widths[i] = DistCentroids[i]/Number[i]*self.h
        
        # calculate the activation Matrix
        G = self._calcAct(features)
         
        # calculate weights (pseudoinverse)
        self.W = np.linalg.lstsq(G.T,target,rcond=None)[0]
         
    def evaluate(self, X):
        G = self._calcAct(X[:-1,:])
        self.y = np.dot(G.T,self.W)
        self.target = X[-1,:]
        return self.y
    
    def meta_find(self,LM,validation_fun):
        n=self.numCenters_list.shape[0]
        m=self.h_list.shape[0]
        p=self.beta_list.shape[0]
        error_array = np.zeros((m,n,p))
        for i,h in enumerate(self.h_list):
            for j,numCenters in enumerate(self.numCenters_list):
                for k,beta in enumerate(self.beta_list):
                    self.h = h
                    self.numCenters = numCenters
                    self.beta = beta
                    error_array[i,j,k] = validation_fun(self,LM)
            
            print("iteration : {}/{}".format(i+1,m))
        i,j,k = np.unravel_index(np.argmin(error_array),(m,n,p))
        self.h = self.h_list[i]
        self.numCenters = self.numCenters_list[j]
        self.beta = self.beta_list[k]
        return (self.h,self.numCenters,self.beta,error_array)
    
    
class mlp2(model):
    def __init__(self):
        model()
        self.nlayers = 2
        self.act_fun = tf.nn.softmax
        self.n_units = 6
        self.act_fun_list = np.array([tf.nn.softmax])
        self.n_units_list = [6,]#np.arange(1,9)
        self.intern_model = None
        
    def train(self,L):
        features = L[:-1,:].T # features DxN
        target = L[-1,:].T #1xN
        #first, we create the tensor_flow intern model
        #architecture of the NN
        self.intern_model=Sequential([
                layers.Dense(self.n_units,activation=self.act_fun,input_shape=(L.shape[0]-1,)),
                layers.Dense(1),#final layer
            ])
        #optimizer we use : stochastic gradient descent
        self.intern_model.compile(loss=['mean_squared_error',],optimizer='sgd',metrics=['mean_squared_error',])
        self.intern_model.summary()
        #then we train the model
        #we do not want to do any validation at that step
        n_epochs=100
        self.intern_model.fit(features,target,epochs=n_epochs,validation_split = 0.2,
                           verbose=1,batch_size=3,shuffle=True)
        
        
        #history = (if needed)
    def evaluate(self,T):
        features = T[:-1,:].T # features DxN
        self.y=self.intern_model.predict(features)
        self.target = T[-1,:].T
        return self.y
    
    def meta_find(self,LM,validation_fun):
        n=len(self.act_fun_list)
        m=len(self.n_units_list)
        error_array = np.zeros((n,m))
        for i,act_fun in enumerate(self.act_fun_list):
            for j,n_units in enumerate(self.n_units_list):
                self.act_fun=act_fun
                self.n_units = n_units
                error_array[i,j] = validation_fun(self,LM)
            
            print("iteration : {}/{}".format(j+1,m))
        i,j = np.unravel_index(np.argmin(error_array),(m,n))
        self.act_fun = self.act_fun_list[i]
        self.n_units = self.n_units_list[j]
        return (self.act_fun,self.n_units,error_array)
