import numpy as np

class model_linear:
    def __init__(self,D):
        self.w = np.zeros((D+1,))
    
    def _bias(self,X):
        N = X.shape[1]
        return np.concatenate((np.ones((1,N)),X),axis=0)
    
    def train(self,L):
        features = self._bias(L[:-1,:]) # features DxN
        target = L[-1,:]
        self.w = np.linalg.lstsq(features.T,target,rcond=None)[0] # solve by minimizing norm2
    
    def error(self,T):
        return np.sqrt(np.mean((self.eval(T)-T[-1,:])**2))
    
    def eval(self,T):
        X = self._bias(T[:-1,:])
        return (self.w).T@X
    
class model_RBFN:
    #This code has largely been inspired by a the code written by Thomas Rückstieß
    #He can be found at the following address :-
    #http://www.rueckstiess.net/research/snippe-ts/show/72d2363e (18-12-11, 17:30)
    def __init__(self, D, numCenters,h):
        self.h = h #smoothing factor
        self.numCenters = numCenters
        self.centers = np.zeros((D,numCenters)) #c_i
        self.widths = np.zeros((numCenters,)) #sigma_i
        self.W = np.zeros((numCenters,)) #w_i
         
    def _basisfunc(self, c, d):
        return exp(-self.beta * norm(c-d)**2)
     
    def _calcAct(self, X):
        # calculate activations of RBFs
        N=X.shape[1]
        G = np.zeros((self.numCenters,N))
        for indexC in range(self.numCenters):
            G[indexC,:]=np.exp(-1/(2*self.widths[indexC])*np.sum((np.array([self.centers[:,indexC] for i in range(N)]).T-X)**2,axis=0))
        return G
     
    def train(self,L):
        N=L.shape[1]
        features = L[:-1,:] # features DxN
        target = L[-1,:] #1xN
        
        # choose random center vectors from training set
        rnd_idx = np.random.permutation(N)[:self.numCenters]
        self.centers = np.array([features[:,i] for i in rnd_idx]).T
        
        # computes widths
        Number = np.zeros((self.numCenters,))
        DistCentroids = np.zeros((self.numCenters,))
        for i in range(N): #for each point, determine its centroïd
            Dist = np.sum((self.centers-np.array([features[:,i] for k in range(self.numCenters)]).T)**2,axis=0)
            indexMin=np.argmin(Dist)
            #and then add the distance
            DistCentroids[indexMin]+= Dist[indexMin]      
            Number[indexMin]+=1
        #mean of the distances for all the centroids
        self.widths = DistCentroids/Number*self.h
        
        # calculate the activation Matrix
        G = self._calcAct(features)
         
        # calculate weights (pseudoinverse)
        self.W = np.linalg.lstsq(G.T,target,rcond=None)[0]
         
    def eval(self, X):
        G = self._calcAct(X[:-1,:])
        Y = np.dot(G.T, self.W)
        return Y
    
    def error(self,T):
        return np.sqrt(np.mean((self.eval(T)-T[-1,:])**2))
    
    
    
    
    
    
        