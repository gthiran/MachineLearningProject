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
X=dataset[:,shuffled_index]
#myplt.plt_dataset(dataset)
       
print("""\n# =============================================================================
# feature selection
# =============================================================================\n""")
X,mean_x,std_x = fsm.normalize(X)
##LM_U,LM_sing = fsm.pca(LM[:-1,:])
##print("the singular values of x are : ")
##print(LM_sing)
##print()
##
##LM_corr = fsm.correlation(LM)
##print("the correlations between inputs and output are : ")
##print(LM_corr)

Q = 8
##LM_selected=LM# = np.concatenate((LM_U[:,:Q].T@LM[:-1,:],np.array([LM[-1,:]])),axis=0)
##T_selected=T#  = np.concatenate((LM_U[:,:Q].T@T[:-1,:],np.array([T[-1,:]])),axis=0)

print("""\n# =============================================================================
# linear model
# =============================================================================\n""")
m_lin = model.linear_regression(Q)
error_lin = vm.kfold(m_lin,X,method_meta=vm.kfold)*std_x[-1]
print('linear generalization error = ',error_lin)

print("""\n# =============================================================================
# knn model
# =============================================================================\n""")

my_knn = model.knn()
error_knn = vm.kfold(my_knn,X,method_meta=vm.kfold)*std_x[-1]
print('knn generalization error = ',error_knn)

print("""\n# =============================================================================
# rbfn model
# =============================================================================\n""")
my_rbfn = model.rbfn(Q)
error_rbfn = vm.kfold(my_rbfn,X,method_meta=vm.kfold)*std_x[-1]
print('rbfn generalization error = ',error_rbfn)