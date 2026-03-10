import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from ucimlrepo import fetch_ucirepo 

def get_adult_data(as_df=False):
    """
    Parse the entire dataset of adult
    """
    #df = pd.read_csv("./data/adult.csv", )
    adult_dataset = fetch_ucirepo(id=2)
    df = adult_dataset.data.original
    df = df.dropna()
    df = df.replace({'?':np.nan}).dropna()
    #df["income"] = df["income"].map({'<=50K': 0, '>50K': 1})
    df["income"] = df["income"].replace({'<=50K.': '<=50K', '>50K.': '>50K'})
    df["income"] = df["income"].map({'<=50K': 0, '>50K': 1})
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
    y = df["age"] #target
    df = df.drop("age", axis=1)
    # hot encode categorical variables
    S = df['sex']
    df = df.drop('sex', axis=1)
    df = drop_str(df)
    log_numeric_features(df)
    X = df.to_numpy() #features
    
    if as_df: #for comparing with agarwal
        return df, S, y
    else:
        return X, S, y


def get_communities_data(as_df=False):
    communities_and_crime = fetch_ucirepo(id=183)
    df = communities_and_crime.data.original
    df = df.fillna(0)

    cols_to_drop = ['communityname', 'state', 'county', 'community', 'fold']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    sens_attrs = ['racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp']
    df['race'] = df[sens_attrs].idxmax(axis=1) #creating a new column based on ethnicity
    df = df.drop(columns=sens_attrs)

    df = df.drop(df[df['ViolentCrimesPerPop']==0].index)
    y = df['ViolentCrimesPerPop'] #target
    df = df.drop('ViolentCrimesPerPop', axis=1)

    #S=1 for white and S=0 for non-white
    mapping = {'racePctWhite':1, 'racepctblack':0, 'racePctAsian':0, 'racePctHisp':0} 
    S = df['race'].map(mapping) 
    df = df.drop('race', axis=1)
    
    X_crime = df.replace('?', np.nan)
    X_crime = X_crime.apply(pd.to_numeric, errors='coerce')
    imputer = SimpleImputer(strategy='mean')
    X_crime_clean = pd.DataFrame(imputer.fit_transform(X_crime), columns=X_crime.columns)
    
    if as_df:
        return X_crime_clean, S, y
    else:
        return X_crime_clean.to_numpy(), S, y
    
def get_frequencies(S):
    p = []
    for p_s in sorted(S.value_counts(1)):
        p.append(p_s)
    return p 
     
def drop_str(df):
    cols = df.columns
    for c in cols:
        if isinstance(df[c][1], str):
            column = df[c]
            df = df.drop(c, axis=1)
    return df

def log_numeric_features(df):
    cols = df.columns
    for c in cols:
        column =df[c]
        unique_values = list(set(column))
        n = len(unique_values)
        if n > 2:
            df[c] = np.log(1 + df[c])

