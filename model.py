import numpy as np
import vq_methods as vqm

class model:
    def __init__(self):
        self.target = None
        self.y = None
        
    def error(self):
        return np.sqrt(np.mean((self.y-self.target)**2))
    
class linear_regression(model):
    def __init__(self,D):
        model()
        self.w = np.zeros((D,))
    
    def _bias(X):
        return np.concatenate((np.ones((1,X.shape[1])),X),axis=0)
    
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
    
    def meta_find(self,LM,validation_fun):
        pass
    
class knn(model):
    def __init__(self):
        model()
        self.k = 4
        self.learning_data = None
        self.learning_target = None
        self.k_list= np.arange(1,11)
        
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
        self.h = 500                              #smoothing factor
        self.numCenters = 5
        self.alpha = 1
        self.beta = 1
        self.epochs = 10
        
        self.centers = np.zeros((D,self.numCenters)) #c_i
        self.widths = np.zeros((self.numCenters,))   #sigma_i
        self.W = np.zeros((self.numCenters,))        #w_i
        self.h_list= np.linspace(1000,10000,11)
        self.numCenters_list=np.arange(31,41)
        self.beta_list = np.array([0.5,1,2,4])
     
    def _calcAct(self, X):
        # calculate activations of RBFs
        N=X.shape[1]
        G = np.zeros((self.numCenters,N))
        for indexC in range(self.numCenters):
            s = np.sum((np.vstack(self.centers[:,indexC])-X)**2,axis=0)
            G[indexC,:]=np.exp(-1/(2*self.widths[indexC])*s)
        return G
     
    def train(self,L):
        N=L.shape[1]
        features = L[:-1,:] # features DxN
        target = L[-1,:] #1xN
        
        # choose random center vectors from training set
        self.centers = vqm.fsens_learning(features,self.alpha,self.beta,
                                          self.numCenters,self.epochs)
        
        # computes widths
        Number = np.zeros((self.numCenters,))
        DistCentroids = np.zeros((self.numCenters,))
        for i in range(N): #for each point, determine its centroïd
            Dist = np.sum((self.centers-np.vstack(features[:,i]))**2,axis=0)
            
            indexMin=np.argmin(Dist)
            #and then add the distance
            DistCentroids[indexMin]+= np.sqrt(Dist[indexMin])      
            Number[indexMin]+=1
        #mean of the distances for all the centroids
        self.widths = DistCentroids/Number*self.h
        
        # calculate the activation Matrix
        G = self._calcAct(features)
         
        # calculate weights (pseudoinverse)
        self.W = np.linalg.lstsq(G.T,target,rcond=None)[0]
         
    def evaluate(self, X):
        G = self._calcAct(X[:-1,:])
        self.y = G.T@self.W
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
        
        i,j,k = np.unravel_index(np.argmin(error_array),(m,n,p))
        self.h = self.h_list[i]
        self.numCenters = self.numCenters_list[j]
        self.beta = self.beta_list[k]
        return (self.h,self.numCenters,self.beta,error_array)