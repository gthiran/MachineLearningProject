""" methods for optimizing the meta parameters"""

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
        
        