import numpy as np

def default(model,LM):
    """ separates the LM set in two sets L and M, train on L,
        evaluate on M and 
        return error
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
    """
    #first, shuffle the LM set
    #shuffled_index = np.random.permutation(length_dataset)
    #LM = LM[:,shuffled_index]
    length_dataset = LM.shape[1]
    print(length_dataset)
    if len(is_leave_one_out)==1:
        N=length_dataset
    else:
        N = 3
        
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