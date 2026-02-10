import numpy as np
import ot 
from ucimlrepo import fetch_ucirepo 


def read_adult_dataset():
    # fetch dataset 
    adult = fetch_ucirepo(id=2) 
    
    # data (as pandas dataframes) 
    X = adult.data.features
    S = X['sex'] #sensitive attribute
    X = X.drop(columns=['sex'])
    y = adult.data.targets 
    
    #metadata 
    #print(adult.metadata) 
    
    # variable information 
    #print(adult.variables) 

    return X,S,y

read_adult_dataset()
