import numpy as np
import vq_methods as vqm

class model:
    def __init__(self):
        self.target = None
        self.y = None
        
    def error(self):
        return np.sqrt(np.mean((self.y-self.target)**2))
    
class linear_regression(model):
    def _bias(X):
        N = X.shape[1]
        return np.concatenate((np.ones((1,N)),X),axis=0)
    
    def train(self,L):
        features = linear_regression._bias(L[:-1,:]) # features DxN
        target = L[-1,:]
        self.w = np.linalg.lstsq(features.T,target,rcond=None)[0] # solve by minimizing norm2
    
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
        
    def train(self,L):
        self.learning_data = L[:-1,:]
        self.learning_target = L[-1,:]
        
    def evaluate(self,T):
        y = np.zeros(T[-1,:].shape)
        for i in range(0,T.shape[1]):
            dist_array = np.zeros((self.learning_data.shape[1],))
            for j in range(0,self.learning_data.shape[1]):
                dist_array[j] = np.linalg.norm(T[:-1,i]-self.learning_data[:,j],ord=2,axis=0)
            
            min_index = np.argsort(dist_array)
            y[i] = np.mean(self.learning_target[min_index[:self.k+1]])
        
        self.target = T[-1,:]
        self.y = y
        return self.y
    
    def meta_find(self,LM,validation_fun,k_list):
        
        error_array = np.zeros(k_list.shape)
        for index,k in enumerate(k_list):
            self.k = k
            error_array[index] = validation_fun(self,LM)
            
        best_k = k_list[np.argmin(error_array)]
        self.k = best_k
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
     
    def _calcAct(self, X):
        # calculate activations of RBFs
        N=X.shape[1]
        G = np.zeros((self.numCenters,N))
        for indexC in range(self.numCenters):
            G[indexC,:]=np.exp(-1/(2*self.widths[indexC])*np.sum((np.array(
                    [self.centers[:,indexC] for i in range(N)]).T-X)**2,axis=0))
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
            Dist = np.sum((self.centers-np.array([features[:,i] for k in 
                                                  range(self.numCenters)]).T)**2,axis=0)
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
        self.y = np.dot(G.T, self.W)
        self.target = X[-1,:]
        return self.y
    
    def meta_find(self,LM,validation_fun,h_list,numCenters_list):
        
        n=numCenters_list.shape[0]
        m=h_list.shape[0]
        error_array = np.zeros((m,n))
        for i,h in enumerate(h_list):
            for j,numCenters in enumerate(numCenters_list):
                self.h = h
                self.numCenters = numCenters
                error_array[i,j] = validation_fun(self,LM)
            
        i,j = np.unravel_index(np.argmin(error_array),(m,n))
        self.h = h_list[i]
        self.numCenters = numCenters_list[j]
        return (self.h,self.numCenters,error_array)