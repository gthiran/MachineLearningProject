import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import model

X1 = pd.read_csv("X1_t1.csv")
data = X1.values.T
learning_set = data[:,:251]
test_set = data[:,251:]

mymodel_linear = model.model_linear(data.shape[0])
mymodel_linear.train(learning_set)
e = mymodel_linear.error(test_set)
y = mymodel_linear.eval(test_set)

print('error on test set = ',e)
plt.figure()
plt.plot(test_set[-1,:],y,'.')
plt.show()