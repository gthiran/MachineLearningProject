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
    
    
    
    
    
    
        