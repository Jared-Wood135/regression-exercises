# =======================================================================================================
# Table of Contents START
# =======================================================================================================

'''
1. Orientation
2. Imports
3. select_kbest
4. rfe
'''

# =======================================================================================================
# Table of Contents END
# Table of Contents TO Orientation
# Orientation START
# =======================================================================================================

'''
The purpose of this file is to create functions for ease of use when conducting feature engineering
related processes.
'''

# =======================================================================================================
# Orientation END
# Orientation TO Imports
# Imports START
# =======================================================================================================

# General imports (Vectorization, dataframe, visualization)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn specific imports
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

# =======================================================================================================
# Imports END
# Imports TO select_kbest
# select_kbest START
# =======================================================================================================

def select_kbest(predictors, target, k_features):
    '''
    Takes in a predictors and target dataframes as well as how many features (column names) you want
    to be selected.

    INPUT:
    predictors = Dataframe with ONLY the predictor columns and their respective values. MUST BE INT, FLOAT.
    target = Dataframe with ONLY the target column and their respective values.
    k_features = The number of top performing features you want to be returned.

    OUTPUT:
    top = A list of column names that are the top performing features.
    '''
    kbest = SelectKBest(f_regression, k=k_features)
    _ = kbest.fit(predictors, target)
    top = predictors.columns[kbest.get_support()].to_list()
    return top

# =======================================================================================================
# select_kbest END
# select_kbest TO rfe
# rfe START
# =======================================================================================================

def rfe(predictors, target, k_features):
    '''
    Takes in a predictors and target dataframes as well as how many features (column names) you want
    to be selected.

    INPUT:
    predictors = Dataframe with ONLY the predictor columns and their respective values. MUST BE INT, FLOAT.
    target = Dataframe with ONLY the target column and their respective values.
    k_features = The number of top performing features you want to be returned.

    OUTPUT:
    top = A dataframe ordered from best to worst performing features.
    '''
    LR = LinearRegression()
    rfe = RFE(LR, n_features_to_select=k_features)
    rfe.fit(predictors, target)
    top = pd.DataFrame({'rfe_ranking' : rfe.ranking_},
                 index=predictors.columns)
    top = pd.DataFrame(top.rfe_ranking.sort_values())
    return top

# =======================================================================================================
# rfe END
# ======================================================================================================