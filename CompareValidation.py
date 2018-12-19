#in order to compare models
#we take the default metaparamters of all the models
#and then we do validation methods
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import model
import validation_methods as vm
import feature_selection_methods as fsm
#
plt.close('all')
print("""\n# =============================================================================
## dataset import 
## =============================================================================\n""")
X1 = pd.read_csv("datasets\X1_t1.csv").transpose()
dataset = X1.values
learning_ratio = 0.8
length_dataset = dataset.shape[1]
index_LM = int(learning_ratio*length_dataset)
shuffled_index = np.random.permutation(length_dataset)
dataset,mean_dataset,std_dataset = fsm.normalize(dataset)

model_list=[model.linear_regression(8), model.knn(), model.rbfn(8), model.mlp2()]
method_list=[vm.default,vm.cross_validation,vm.kfold,vm.bootstrap]
Niter=10
Nmodel=len(model_list)
Nmethod = len(method_list)
PredictErrorDef = np.zeros((Niter,Nmodel,Nmethod))
RealErrorDef = np.zeros((Niter,Nmodel))
for iter in range(Niter):
    print('iter : {}/{}'.format(iter+1,Niter))
    shuffled_index = np.random.permutation(length_dataset)
    LM = dataset[:,shuffled_index[:index_LM]]
    T = dataset[:,shuffled_index[index_LM:]]
    for index,mod in enumerate(model_list):
        print('model : {}/{}'.format(index,Nmodel))
        mod.train(LM)
        mod.evaluate(T)
        RealErrorDef[iter,index]=mod.error()
        for i,method in enumerate(method_list):
            print('meth : {}/{}'.format(i,Nmethod))
            PredictErrorDef[iter,index,i]=method(mod,LM)
            

PredictErrorMean = np.mean(PredictErrorDef,axis=0)
RealErrorMean = np.mean(RealErrorDef,axis=0)
ErrorError = np.zeros((Nmodel,Nmethod))
for i in range(Nmethod):
    ErrorError[:,i]=np.abs(RealErrorMean-PredictErrorMean[:,i])
ErrorError=np.array([[0.0265,0.0059172,0.01295,0.0062],[0.0061,0.0077,0.011,0.008],[0.02484,0.00992036,0.02492,0.0055],[0.01479,0.0172,0.0372593,0.0147]]).T
ErrorError = ErrorError.T*std_dataset[-1] #denormalize the error
fig = plt.figure(0)
N = 4
ind = np.arange(N)    # the x locations for the groups
width = 0.35 /2      # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()
p1 = ax.bar(ind - 3*width/2, ErrorError[0,:], width)
p2 = ax.bar(ind - width/2, ErrorError[1,:], width)
p3 = ax.bar(ind + width/2, ErrorError[2,:], width)
p4 = ax.bar(ind + 3*width/2, ErrorError[3,:], width)

plt.ylabel('Error on the generalization Error')
plt.title('Performances of the validation methods')
plt.xticks(ind, ('default', 'cross validation', 'kfold', 'bootstrap'))
plt.legend((p1[0], p2[0],p3[0],p4[0]), ('Linear', 'knn','rbfn','mlp'))

plt.show()
fig.savefig('ErrorVal.png')