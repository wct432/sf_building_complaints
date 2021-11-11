import numpy as np



def get_proportions(subset):
    """
    Used to return the proportions of each class in multi-class labels of 
    train_test_split subsets that have been encoded using sklearn's LabelBinarizer 
    or OneHotEncoder.
	Args:
		subset: The subset to evaluate, ie: get_proportions(y_train).
	Returns:
		A dictionary that contains normalized proportions of each class.
    """
    proportions = {}
    for row in subset:
        for i, x in enumerate(row):
            if i in proportions:
                proportions[i] += x
            else:
                proportions[i] = 1
    #normalize each proportion by diving by the total rows in the subset
    for i in proportions:
        proportions[i] = float(proportions[i] / subset.shape[0])
    return proportions