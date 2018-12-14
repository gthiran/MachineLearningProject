import numpy as np

def pca(X):
    u,s,vh = np.linalg.svd(X)
    return (u,s)

def correlation(X):
    R = np.corrcoef(X,rowvar=True)
    return R[-1,:-1]
    

def normalize(X):
    u = np.mean(X,axis=1)
    s = np.std(X,axis=1)
    e = np.ones((1,X.shape[1]))
    return ((X-np.outer(u,e))/(np.outer(s,e)),u,s)

def denormalize(Z,u,s):
    return Z*s+u


        
    
        
    