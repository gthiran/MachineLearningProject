import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import model

X1 = pd.read_csv("X1_t1.csv")
data = X1.values.T
learning_set = data[:,:251]
test_set = data[:,251:]

""" linear model"""
mymodel_linear = model.model_linear()
mymodel_linear.train(learning_set)
y = mymodel_linear.eval(test_set)
e = mymodel_linear.error()

print('linear error = ',e)
plt.figure()
plt.plot(test_set[-1,:],y,'.')

""" knn model"""
k=4
mymodel_knn = model.model_knn(k)
mymodel_knn.train(learning_set)
y = mymodel_knn.eval(test_set)
e = mymodel_knn.error()

print('knn error = ',e)
plt.figure()
plt.plot(test_set[-1,:],y,'.')
