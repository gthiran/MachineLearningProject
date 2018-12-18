import numpy as np
""" General description:
    All these methods want to obtain the generalization error of the model
    To do so, we can apply a validation, cross-validation-k-fold,leave-one-out or bootstrap
    All these methods separate the X set into two sets : set1 and set2
        if method_meta is given, that means we want to evaluate the model
            Therefore, set1 corresponds to (learning + meta) set and set2 corresponds to test set
            First, we find the meta optimal with method_meta on (learning+meta) set
            Then we train on set1 and evaluate on set 2
            to obtain a generalization error of the model with optimal meta_parameters
        else, that means we are interested in testing the model with given meta_parameters
            #we thus train on set1 and evaluate on set2
            #to obtain the error of generalization for these meta_parameters
    Attention, all the methods suppose X is already shuffled
"""
def default(model,X,method_meta=None):
    """
       General description at the beginning of the file
       Validation method:
       Separate X in set1 and set2 once, at random
       Return the generalization error
    """
    set_ratio = 0.75 #ratio between set1 and the original dataset
    length_dataset = X.shape[1] #size of the original dataset
    index_set1 = int(set_ratio*length_dataset) #index of the sepration
    #since we suppose X is shuffled, we can just separate X 
    set1 = X[:,:index_set1]
    set2 = X[:,index_set1:]
    if method_meta is not None:#if we have to find the optimal meta_parameters
        model.meta_find(set1,method_meta)
    model.train(set1)#train on set1
    model.evaluate(set2)#evaluate on set2
    return model.error()#error on set2

def cross_validation(model,X,method_meta=None):
    """
       General description at the beginning of the file
       Cross-Validation method:
       Separate X in set1 and set2 several times, at random
       Return the mean generalization error
    """
    Ncross = 10 #number of repetition
    set_ratio = 0.75
    length_dataset = X.shape[1]
    index_set1 = int(set_ratio*length_dataset)

    error_array = np.zeros((Ncross,))
    for i in range(Ncross):
        #first, shuffle the set1 set
        shuffled_index = np.random.permutation(length_dataset)
        X = X[:,shuffled_index]
        #then define new set1 and set2
        set1 = X[:,:index_set1]
        set2 = X[:,index_set1:]
        if method_meta is not None:#if we have to find the optimal meta_parameters
            model.meta_find(set1,method_meta)
            print("cross_validation : iter = {}/{}".format(i+1,Ncross))
        #then, train the model on set1
        model.train(set1)
        #and evaluate on set2
        model.evaluate(set2)
        error_array[i]=model.error()#generalization error on set 2
    
    return(np.mean(error_array))

def kfold(model,X,is_leave_one_out = False,method_meta = None):
    """
       General description at the beginning of the file
       kfold method:
       Separate X in N sets, set1 = (N-1) subsets and set2 = 1 subset
       Return the mean generalization error on the N subsets
       if is_leave_one_out, apply kfold with N= length(X)
    """
    length_dataset = X.shape[1]
    if is_leave_one_out: #if we do leave_one_out
        N=length_dataset
    else:
        N = 10
        
    length_block =int(length_dataset/N)
    error_array = np.zeros((N,))
    for i in range(N):
        #define the index of start and end of set1
        indexStart = i*length_block
        indexEnd = (i+1)*length_block
        index_set1=np.concatenate((np.arange(0,indexStart) ,np.arange(indexEnd,length_dataset)))
        index_set2=np.arange(indexStart,indexEnd)
        #then define new set1 and set2
        set1 = X[:,index_set1]
        set2 = X[:,index_set2]
        if method_meta is not None:#if we have to find the optimal meta_parameters
            model.meta_find(set1,method_meta)
            print("kfold (length_block : {}) : iter = {}/{}".format(length_block,i+1,N))
        #then, train the model on set1
        model.train(set1)
        #and evaluate on set2
        model.evaluate(set2)
        error_array[i]=model.error() #generalization error on set 2
        
    return(np.mean(error_array))

def leave_one_out(model,X,method_meta=None):
    """
       General description at the beginning of the file
       leave-one-out method:
       k_fold method with N = length(X)
    """
    if method_meta is not None:#if we have to find the optimal meta_parameters
        return(kfold(model,X,True,method_meta))
    else: #else
        return(kfold(model,X,True))
    
def bootstrap(model,X,method_meta=None):
    """ apply the bootstrap method on X
    """
    ratio_Subsets = 0.75
    Nboot=10 #number of bootstrap iteration
    length_dataset = X.shape[1]
    size_Subsets = int(ratio_Subsets*length_dataset)
    
    opt_array = np.zeros((Nboot,))#array for the optimisms
    for i in range(Nboot):
        #first, draw with replacement from X
        #to obtain X*
        shuffled_index=np.random.choice(length_dataset, size_Subsets, replace=True)
        set1 = X[:,shuffled_index]
        if method_meta is not None:#if we have to find the optimal meta_parameters
            model.meta_find(set1,method_meta)
            print("bootstrap : iter = {}/{}".format(i+1,Nboot))
        #then, train the model on set1
        model.train(set1)
        #and evaluate on set1 and X
        model.evaluate(set1)
        Eset1set1=model.error()
        model.evaluate(X)
        Eset1X=model.error()
        #compute optimism:
        opt_array[i] = Eset1X-Eset1set1
    
    if method_meta is not None:#if we have to find the optimal meta_parameters
            model.meta_find(X,method_meta)
    #train the model on X
    model.train(X)
    #and evaluate on X
    model.evaluate(X)
    EXX=model.error()
               
    return EXX+np.mean(opt_array)
    