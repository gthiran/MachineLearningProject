import numpy as np

class model:
    def __init__(self):
        self.target = None
        self.y = None
        
    def error(self):
        return np.sqrt(np.mean((self.y-self.target)**2))
    
class model_linear(model):
    def _bias(X):
        N = X.shape[1]
        return np.concatenate((np.ones((1,N)),X),axis=0)
    
    def train(self,L):
        features = model_linear._bias(L[:-1,:]) # features DxN
        target = L[-1,:]
        self.w = np.linalg.lstsq(features.T,target,rcond=None)[0] # solve by minimizing norm2
    
    def eval(self,T):
        X = model_linear._bias(T[:-1,:])
        self.target = T[-1,:]
        self.y = (self.w).T@X
        return self.y
    
class model_knn(model):
    def __init__(self,K):
        self.k = K
        
    def train(self,L):
        self.learning_data = L[:-1,:]
        self.learning_target = L[-1,:]
        
    def eval(self,T):
        y = np.zeros(T[-1,:].shape)
        for i in range(0,T.shape[1]):
            dist_array = np.zeros((self.learning_data.shape[1],))
            for j in range(0,self.learning_data.shape[1]):
                dist_array[j] = np.linalg.norm(T[:-1,i]-self.learning_data[:,j],ord=2,axis=0)
            
            id = np.argsort(dist_array)
            y[i] = np.mean(self.learning_target[id[:self.k+1]])
        
        self.target = T[-1,:]
        self.y = y
        return self.y
