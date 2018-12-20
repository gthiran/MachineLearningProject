import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import model
import validation_methods as vm
import feature_selection_methods as fsm

plt.close('all')
print("""\n# =============================================================================
# dataset import
# =============================================================================\n""")
X1 = pd.read_csv("datasets\X1_t1.csv").transpose()
X2 = pd.read_csv("datasets\X2.csv").transpose()

dataset = X1.values
dataset2 = X2.values

learning_ratio = 1
length_dataset = dataset.shape[1]
index_LM = int(learning_ratio*length_dataset)
shuffled_index = np.random.permutation(length_dataset)

dataset,mean_dataset,std_dataset = fsm.normalize(dataset)
dataset2,mean_dataset2,std_dataset2 = fsm.normalize(dataset2)

print("""\n# =============================================================================
# feature selection
# =============================================================================\n""")
LM = dataset[:,shuffled_index[:index_LM]]
T = dataset[:,shuffled_index[index_LM:]]

LM_U,LM_sing = fsm.pca(LM[:-1,:])
LM_U2,LM_sing2 = fsm.pca(dataset2)
print("the singular values of x are : ")
print(LM_sing)
print()

LM_corr = fsm.correlation(LM)
print("the correlations between inputs and output are : ")
print(LM_corr)

Q = 7
LM_selected= np.concatenate((LM_U[:,:Q].T@LM[:-1,:],LM[-1,:].reshape(1,index_LM)),axis=0)
T_selected = np.concatenate((LM_U[:,:Q].T@T[:-1,:],T[-1,:].reshape(1,length_dataset-index_LM)),axis=0)

X2_selected = np.concatenate((LM_U2[:,:Q].T@dataset2,np.zeros((1,dataset2.shape[1]))),axis=0)

print("""\n# =============================================================================
# mlp2 model
# =============================================================================\n""")
my_mlp = model.mlp2()

act_funct,n_nodes,learning_rate,error_array_mlp2 = my_mlp.meta_find(LM_selected,vm.bootstrap)

error_array_mlp2=error_array_mlp2*std_dataset[-1]
print("act_opt = {}".format(act_funct))
print("n_nodes_opt = {}".format(n_nodes))
print("learning rate = {}".format(learning_rate))

print("error_array = ")
print(error_array_mlp2)

print("""\n# =============================================================================
# final simulation
# =============================================================================\n""")

my_mlp.train(LM_selected)

Y2_gr_M = my_mlp.evaluate(X2_selected)*std_dataset[-1]+mean_dataset[-1]

print("done")