import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import model
import validation_method as vm


X1 = pd.read_csv("X1_t1.csv").transpose()
dataset = X1.values





# find meta params
#nshuffle = 3
#k = np.arange(1,51)
#e_knn = np.zeros(k.shape)
#for i in range(len(k)):
#    
#    for j in range(nshuffle):
#        shuffled_index = np.random.permutation(length_dataset)
#        
#        learning_set = dataset[:,shuffled_index[:index_learning_set]]
#        meta_set = dataset[:,shuffled_index[index_learning_set:index_meta_set]]
#        test_set = dataset[:,shuffled_index[index_meta_set:]]
#        
#        m_knn = model.knn(k[i])
#        m_knn.train(learning_set)
#        y = m_knn.evaluate(meta_set)
#        e_knn[i] += m_knn.error()
#    
#    print("k = {}".format(k[i]))
#    e_knn[i] /= nshuffle
#    
#plt.figure()
#plt.plot(k,e_knn,linewidth=1.2)
#plt.grid()
#plt.show()
    
learning_ratio = 0.75
length_dataset = dataset.shape[1]
index_LM = int(learning_ratio*length_dataset)
shuffled_index = np.random.permutation(length_dataset)
        
LM = dataset[:,shuffled_index[:index_LM]]
test_set = dataset[:,shuffled_index[index_LM:]]

#"""linear model"""
#m_lin = model.linear_regression()
#m_lin.train(learning_set)
#y_lin = m_lin.evaluate(test_set)
#e_lin = m_lin.error()
#
#print('linear error = ',e_lin)
#plt.figure()
#plt.plot(test_set[-1,:],y_lin,'.')
#plt.show()

#""" knn model"""
#k_list = np.arange(1,11)
#my_knn = model.knn()
#k_opt,error_array_knn = my_knn.meta_find(LM,vm.default,k_list)
#
#print("=== knn ===")
#print("k_opt = {}".format(k_opt))
#print("error_array = ")
#print(error_array_knn)
#
#my_knn.train(LM)
#y = my_knn.evaluate(test_set)
#e_knn = my_knn.error()
#
#print('error = ',e_knn)
#plt.figure()
#plt.plot(test_set[-1,:],y,'.')
#plt.show()

""" rbfn model """
h_list = np.linspace(450,550,11)
numCenters_list = np.arange(10,31)
my_rbfn = model.rbfn(8)
h_opt,numCenters_opt, error_array_rbfn = my_rbfn.meta_find(
        LM,vm.default,h_list,numCenters_list)

print("== rbfn ===")
print("h_opt = {}".format(h_opt))
print("numCenters_opt = {}".format(numCenters_opt))
print("error_array = ")
print(error_array_rbfn)

my_rbfn.train(LM)
y = my_rbfn.evaluate(test_set)
e_rbfn = my_rbfn.error()

print('error = ',e_rbfn)
plt.figure()
plt.plot(test_set[-1,:],y,'.')
plt.show()