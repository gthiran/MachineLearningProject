import numpy as np

""" vq methods """
def init_centroids(X,nCentroids):
    N=X.shape[1]
    index = np.random.permutation(N)
    return X[:,index[:nCentroids]]

def fsens_learning(X,alpha,beta,nCentroids,epochs):
    X_c = init_centroids(X,nCentroids)
    U = np.ones((nCentroids,))
    
    for i in np.arange(epochs):
        for j in np.arange(X.shape[1]):
            d = U*np.linalg.norm(X_c-np.array([X[:,j]]).T,ord=2,axis=0)
            index = np.argmin(d)
            U[index]+=1
            X_c[:,index]+=alpha*(X[:,j]-X_c[:,index])
        
        alpha*=beta/(alpha+beta)
    
    return X_c

