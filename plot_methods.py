import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})

def plt_compare_results(y_lin,y_knn,y_rbfn,y_mlp,std_dataset,mean_dataset,T):
    xx = np.linspace(0,80,200)
    # Four axes, returned as a 2-d array
    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].plot(T[-1,:]*std_dataset[-1]+mean_dataset[-1],y_lin,'.')
    axarr[0, 0].plot(xx,xx,'k')
    axarr[0, 0].set_title('Linear : y vs t')
    axarr[0, 0].set_xlabel('target : t')
    axarr[0, 0].set_ylabel('model output : y')

    axarr[0, 1].plot(T[-1,:]*std_dataset[-1]+mean_dataset[-1],y_knn,'.')
    axarr[0, 1].plot(xx,xx,'k')
    axarr[0, 1].set_title('knn : y vs t')
    axarr[0, 1].set_xlabel('target : t')
    axarr[0, 1].set_ylabel('model output : y')

    axarr[1, 0].plot(T[-1,:]*std_dataset[-1]+mean_dataset[-1],y_rbfn,'.')
    axarr[1, 0].plot(xx,xx,'k')
    axarr[1, 0].set_title('rbfn : y vs t')
    axarr[1, 0].set_xlabel('target : t')
    axarr[1, 0].set_ylabel('model output : y')

    axarr[1, 1].plot(T[-1,:]*std_dataset[-1]+mean_dataset[-1],y_mlp,'.')
    axarr[1, 1].plot(xx,xx,'k')
    axarr[1, 1].set_title('mlp : y vs t')
    axarr[1, 1].set_xlabel('target : t')
    axarr[1, 1].set_ylabel('model output : y')

def plt_dataset(dataset):
    plt.figure()
    p=420
    for i in range(1,9):
        plt.subplot(p+i)
        plt.plot(dataset[i-1,:],dataset[-1,:],'.',label="X{}".format(i))
        plt.legend(loc=1)
        plt.grid()

    plt.show()

def plt_knn_meta(my_knn,error_array_knn):
    plt.figure()
    plt.plot(my_knn.k_list,error_array_knn,'-k')
    plt.plot(my_knn.k,np.min(error_array_knn),'.r',markersize=20)
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.grid()

def plt_rbfn_meta(my_rbfn,error_array_rbfn):
    numCenters,h = np.meshgrid(my_rbfn.numCenters_list,my_rbfn.h_list)
    e=np.log10(error_array_rbfn)
    plt.figure()
    cs=plt.contourf(numCenters,h,e,10,cmap=plt.cm.jet)
    cbar=plt.colorbar(cs)
    cbar.ax.set_ylabel('error (log10)')
    plt.xlabel('num centers')
    plt.ylabel('h')
    plt.grid()
    plt.show()
