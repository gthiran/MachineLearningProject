import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import model
import validation_methods as vm
import feature_selection_methods as fsm
import plot_methods as myplt

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

myplt.plt_dataset(dataset)
       
print("""\n# =============================================================================
# feature selection
# =============================================================================\n""")
LM = dataset[:,shuffled_index[:index_LM]]
T = dataset[:,shuffled_index[index_LM:]]
LM[:-1,:],mean_LM,std_LM = fsm.normalize(LM[:-1,:])
T[:-1,:],mean_T,std_T = fsm.normalize(T[:-1,:])

LM_U,LM_sing = fsm.pca(LM[:-1,:])
print("the singular values of x are : ")
print(LM_sing)
print()

LM_corr = fsm.correlation(LM)
print("the correlations between inputs and output are : ")
print(LM_corr)

Q = 6
LM_selected = np.concatenate((LM_U[:,:Q].T@LM[:-1,:],np.array([LM[-1,:]])),axis=0)
T_selected  = np.concatenate((LM_U[:,:Q].T@T[:-1,:],np.array([T[-1,:]])),axis=0)

print("""\n# =============================================================================
# linear model
# =============================================================================\n""")
m_lin = model.linear_regression(Q)
m_lin.train(LM_selected)
y_lin = m_lin.evaluate(T_selected)
e_lin = m_lin.error()

print('linear error = ',e_lin)
myplt.plt_compare_results(y_lin,T[-1,:],"LINEAR")

print("""\n# =============================================================================
# knn model
# =============================================================================\n""")
k_list = np.arange(1,11)
my_knn = model.knn()
k_opt,error_array_knn = my_knn.meta_find(LM_selected,vm.kfold,k_list)

print("k_opt = {}".format(k_opt))
print("error_array = ")
print(error_array_knn)

my_knn.train(LM_selected)
y_knn = my_knn.evaluate(T_selected)
e_knn = my_knn.error()

print('error = ',e_knn)
myplt.plt_compare_results(y_knn,T[-1,:],"KNN")

print("""\n# =============================================================================
# rbfn model
# =============================================================================\n""")
h_list = np.linspace(1000,10000,11)
numCenters_list = np.arange(31,41)
my_rbfn = model.rbfn(Q)
h_opt,numCenters_opt, error_array_rbfn = my_rbfn.meta_find(
        LM_selected,vm.default,h_list,numCenters_list)

print("h_opt = {}".format(h_opt))
print("numCenters_opt = {}".format(numCenters_opt))
print("error_array = ")
print(error_array_rbfn)

my_rbfn.train(LM_selected)
y_rbfn = my_rbfn.evaluate(T_selected)
e_rbfn = my_rbfn.error()

print('error = ',e_rbfn)
myplt.plt_compare_results(y_rbfn,T[-1,:],"RBFN")