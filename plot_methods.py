import numpy as np
import matplotlib.pyplot as plt

def plt_compare_results(Y,T,title):
    x = np.arange(T.size)
    plt.figure()
    plt.plot(np.concatenate((np.vstack(x),np.vstack(x)),axis=1).T,
             np.concatenate((np.vstack(T),np.vstack(Y)),axis=1).T,'-k')
    plt.plot(x,T,'.r')
    plt.plot(x,Y,'.b')
    plt.grid
    plt.xlabel('Sample')
    plt.ylabel('Output')
    plt.title(title)
    plt.show()
    
def plt_dataset(dataset):
    plt.figure()
    p=420
    for i in range(1,9):
        plt.subplot(p+i)
        plt.plot(dataset[i-1,:],dataset[-1,:],'.',label="X{}".format(i))
        plt.legend()
        plt.grid()
        
    plt.show()