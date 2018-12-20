import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import model
import validation_methods as vm
import feature_selection_methods as fsm
import plot_methods as my_plt

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
my_plt.plt_dataset(dataset)

print("""\n# =============================================================================
# feature selection
# =============================================================================\n""")
X,mean_x,std_x = fsm.normalize(X)
X_U,X_sing = fsm.pca(X[:-1,:])
print("the singular values of x are : ")
print(X_sing)
print()

LM_corr = fsm.correlation(X)
print("the correlations between inputs and output are : ")
print(LM_corr)

Q = 7
#Xindex = np.array([1,2,4,5,6,7,8,9])-1
X = np.concatenate((X_U[:,:Q].T@X[:-1,:],np.array([X[-1,:]])),axis=0)
print(X.shape)

print("""\n# =============================================================================
# linear model
# =============================================================================\n""")
m_lin = model.linear_regression(Q)
error_lin = vm.bootstrap(m_lin,X)*std_x[-1]
print('linear generalization error = ',error_lin)

print("""\n# =============================================================================
# knn model
# =============================================================================\n""")

my_knn = model.knn()
error_knn = vm.bootstrap(my_knn,X,method_meta=vm.bootstrap)*std_x[-1]
print('knn generalization error = ',error_knn)

print("""\n# =============================================================================
# rbfn model
# =============================================================================\n""")
my_rbfn = model.rbfn(Q)
error_rbfn = vm.bootstrap(my_rbfn,X,method_meta=vm.bootstrap)*std_x[-1]
print('rbfn generalization error = ',error_rbfn)

print("""\n# =============================================================================
# mlp model
# =============================================================================\n""")
my_mlp = model.mlp2()
error_mlp = vm.default(my_mlp,X,method_meta=vm.bootstrap)*std_x[-1]
print('mlp generalization error = ',error_mlp)