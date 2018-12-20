import numpy as np
import scipy.special as sp

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

def mutual_infoKSG(X,Y,k):
    #compute the mutual information between x and y
    #x and y are 1D

    size=X.shape
    if len(size)==1 : #dim 1
        #normalize
        N=size[0]
        X=(X-np.mean(X))/np.sqrt(np.sum(np.abs(X - np.mean(X))**2)/(N-1))
        Y=(Y-np.mean(Y))/np.sqrt(np.sum(np.abs(Y - np.mean(Y))**2)/(N-1))
         #for the two sets, we need the minimal distance between any pair of points
        #preallocation
        Dx = np.zeros((N,N))
        Dy = np.zeros((N,N))
        for i in range(N):
            #for each point, compute its distance from all the other points
            Dx[i,:]=np.abs(X[i]-X)
            Dx[i,i]=float('inf') #we don't want to compare the point with itself

            Dy[i,:]=np.abs(Y[i]-Y)
            Dy[i,i]=float('inf')#we don't want to compare the point with itself
    else :
        print('error')
        return

    #now we have to find, for each point, its kth nearest neighbour
    D=np.maximum(Dx,Dy) #maxnorm
    arg=np.argsort(D, axis=0)
    #now, the array is sorted with the index corresponding to the smallest distances on the upper part
    #the k first lines interest ourselves
    indexkth = arg[:k-1,:]
    #vector ex(i,k), with k static
    argsum = np.zeros((N,))
    for i in range(N):
        exik=np.max(Dx[indexkth[:,i],i])*2 #2*
        eyik=np.max(Dy[indexkth[:,i],i])*2 #2*
        #now, for each point, we have to count how many point are in the range
        nx=np.sum(Dx[:,i]<=exik/2,axis=0) #matrices of booleans, 1 si dans le range
        ny=np.sum(Dy[:,i]<=eyik/2,axis=0) #matrices of booleans, 1 si dans le range
       #digamma on these functions
        argsum[i]=sp.digamma(nx)+sp.digamma(ny)
    mutual_info = sp.digamma(N)+sp.digamma(k)-(1/k)-np.mean(argsum)
    return mutual_info


