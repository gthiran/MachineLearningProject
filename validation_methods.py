import numpy as np

def default(model,LM):
    """ separates the LM set in two sets L and M, train on L,
        evaluate on M and 
        return error
        Suppose LM is already shuffled
    """
    learning_ratio = 0.75
    length_dataset = LM.shape[1]
    index_learning_set = int(learning_ratio*length_dataset)
    learning_set = LM[:,:index_learning_set]
    meta_set = LM[:,index_learning_set:]
        
    model.train(learning_set)
    model.evaluate(meta_set)
    return model.error()

def cross_validation(model,LM):
    """ separates the LM set in two sets L and M, train on L,
        evaluate on M and repeat several times (attenion hardcoded parameter)
        return mean error
    """
    Ncross = 10
    learning_ratio = 0.75
    length_dataset = LM.shape[1]
    index_learning_set = int(learning_ratio*length_dataset)

    error_Meta = np.zeros((Ncross,))
    for i in range(Ncross):
        #first, shuffle the LM set
        shuffled_index = np.random.permutation(length_dataset)
        LM = LM[:,shuffled_index]
        #then define new learning and meta sets
        learning_set = LM[:,:index_learning_set]
        meta_set = LM[:,index_learning_set:]
        #then, train the model on the learning set
        model.train(learning_set)
        #and evaluate on the meta set
        model.evaluate(meta_set)
        error_Meta[i]=model.error()
    
    return(np.mean(error_Meta))

def kfold(model,LM,*is_leave_one_out):
    """ separates the LM set in N sets train an N-1 sets,
        and evaluate on the remaining set. 
        return mean error
        suppose
    """
    length_dataset = LM.shape[1]
    if len(is_leave_one_out)==1:
        N=length_dataset
    else:
        N = 5
        
    length_block =int(length_dataset/N);
    error_Meta = np.zeros((N,))
    for i in range(N):
        #define the index of start and end of the validation set
        indexStart = i*length_block
        indexEnd = (i+1)*length_block
        index_learning=np.concatenate((np.arange(0,indexStart) ,np.arange(indexEnd,length_dataset)))
        index_Meta=np.arange(indexStart,indexEnd)
        #then define new learning and meta sets
        learning_set = LM[:,index_learning]
        meta_set = LM[:,index_Meta]
        #then, train the model on the learning set
        model.train(learning_set)
        #and evaluate on the meta set
        model.evaluate(meta_set)
        error_Meta[i]=model.error()
    return(np.mean(error_Meta))

def leave_one_out(model,LM):
    """ separates the LM set in sets train meta sets,
        and evaluate on only one element. 
        return mean error
    """
    return(kfold(model,LM,True))

def bootstrap(model,S):
    """ apply the bootstrap method on S
    """
    ratioSubS = 0.75
    Nboot=10
    length_dataset = S.shape[1]
    size_SubS = int(ratioSubS*length_dataset)
    
    opt_array = np.zeros((Nboot,))
    for i in range(Nboot):
        #first, draw with replacement from S
        #to obtain S*
        shuffled_index=np.random.choice(length_dataset, size_SubS, replace=True)
        subS = S[:,shuffled_index]
        #then, train the model on SubS
        model.train(subS)
        #and evaluate on SubS and S
        model.evaluate(subS)
        EsubSsubS=model.error()
        model.evaluate(S)
        EsubSS=model.error()
        #compute optimism:
        opt_array[i] = EsubSS-EsubSsubS
    
    #train the model on S
    model.train(S)
    #and evaluate in S
    model.evaluate(S)
    ESS=model.error()
               
    return ESS+np.mean(opt_array)
    