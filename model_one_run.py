import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import model
import validation_methods as vm
import feature_selection_methods as fsm
import plot_methods as myplt
import time

plt.close('all')
print("""\n# =============================================================================
# dataset import 
# =============================================================================\n""")
X1 = pd.read_csv("datasets\X1_t1.csv").transpose()
dataset = X1.values
learning_ratio = 0.8
length_dataset = dataset.shape[1]
index_LM = int(learning_ratio*length_dataset)
shuffled_index = np.random.permutation(length_dataset)
dataset,mean_dataset,std_dataset = fsm.normalize(dataset)
#myplt.plt_dataset(dataset)
       
print("""\n# =============================================================================
# feature selection
# =============================================================================\n""")
LM = dataset[:,shuffled_index[:index_LM]]
T = dataset[:,shuffled_index[index_LM:]]
#
#LM_U,LM_sing = fsm.pca(LM[:-1,:])
#print("the singular values of x are : ")
#print(LM_sing)
#print()
#
#LM_corr = fsm.correlation(LM)
#print("the correlations between inputs and output are : ")
#print(LM_corr)
#
Q = 8
LM_selected=LM# = np.concatenate((LM_U[:,:Q].T@LM[:-1,:],np.array([LM[-1,:]])),axis=0)
T_selected =T# np.concatenate((LM_U[:,:Q].T@T[:-1,:],np.array([T[-1,:]])),axis=0)
#
#print("""\n# =============================================================================
## linear model
## =============================================================================\n""")
#m_lin = model.linear_regression(Q)
#m_lin.train(LM_selected)
#y_lin = m_lin.evaluate(T_selected)*std_dataset[-1]+mean_dataset[-1]
#e_lin = m_lin.error()*std_dataset[-1]
#
#print('linear error = ',e_lin)
#myplt.plt_compare_results(y_lin,T[-1,:]*std_dataset[-1]+mean_dataset[-1],"LINEAR")
#
#print("""\n# =============================================================================
## knn model
## =============================================================================\n""")
#
#my_knn = model.knn()
#k_opt,error_array_knn = my_knn.meta_find(LM_selected,vm.kfold)
#error_array_knn=error_array_knn*std_dataset[-1]
#print("k_opt = {}".format(k_opt))
#print("error_array = ")
#print(error_array_knn)
#
#my_knn.train(LM_selected)
#y_knn = my_knn.evaluate(T_selected)*std_dataset[-1]+mean_dataset[-1]
#e_knn = my_knn.error()*std_dataset[-1]
#
#print('error = ',e_knn)
#myplt.plt_compare_results(y_knn,T[-1,:]*std_dataset[-1]+mean_dataset[-1],"KNN")

#print("""\n# =============================================================================
## rbfn model
## =============================================================================\n""")
#my_rbfn = model.rbfn(Q)
#h_opt,numCenters_opt,beta_opt, error_array_rbfn = my_rbfn.meta_find(LM_selected,vm.cross_validation)
#error_array_rbfn=error_array_rbfn*std_dataset[-1]
#print("h_opt = {}".format(h_opt))
#print("numCenters_opt = {}".format(numCenters_opt))
#print("beta_opt = {}".format(beta_opt))
#print("error_array = ")
#print(error_array_rbfn)
#
#my_rbfn.train(LM_selected)
#y_rbfn = my_rbfn.evaluate(T_selected)*std_dataset[-1]+mean_dataset[-1]
#e_rbfn = my_rbfn.error()*std_dataset[-1]
#
#print('error = ',e_rbfn)
#myplt.plt_compare_results(y_rbfn,T[-1,:]*std_dataset[-1]+mean_dataset[-1],"RBFN")
print("""\n# =============================================================================
# mlp2 model
# =============================================================================\n""")
my_mlp = model.mlp2()

t1 = time.time()
act_funct,n_nodes,learning_rate, error_array_mlp2 = my_mlp.meta_find(LM_selected,vm.default)
print(time.time()-t1)
error_array_mlp2=error_array_mlp2*std_dataset[-1]
print("act_opt = {}".format(act_funct))
print("n_nodes_opt = {}".format(n_nodes))
print("learning rate = {}".format(learning_rate))

print("error_array = ")
print(error_array_mlp2)
my_mlp.train(LM_selected)

y_mlp = my_mlp.evaluate(T_selected)*std_dataset[-1]+mean_dataset[-1]
e_mlp = my_mlp.error()*std_dataset[-1]

print('error = ',e_mlp)
myplt.plt_compare_results(y_mlp,T[-1,:]*std_dataset[-1]+mean_dataset[-1],"RBFN")
