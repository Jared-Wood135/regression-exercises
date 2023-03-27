# =======================================================================================================
# Table of Contents START
# =======================================================================================================

'''
1. Orientation
2. Imports
3. plot_variable_pairs
4. plot_categorical_and_continuous_vars
'''

# =======================================================================================================
# Table of Contents END
# Table of Contents TO Orientation
# Orientation START
# =======================================================================================================

'''
The purpose of this file is to create functions for exploratory and visualization purposes
'''

# =======================================================================================================
# Orientation END
# Orientation TO Imports
# Imports START
# =======================================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from scipy import stats
from sklearn.model_selection import train_test_split
import os
import wrangle

# =======================================================================================================
# Imports END
# Imports TO plot_variable_pairs
# plot_variable_pairs START
# =======================================================================================================

def plot_variable_pairs(df):
    '''
    Takes in a dataframe and gets all of the float and int dtype columns then 
    returns a histogram, boxplot, and a .describe for each column
    '''
    quantitative_col = df.select_dtypes(include=['float', 'int'])
    for col in quantitative_col:
        sns.histplot(quantitative_col[col])
        plt.title(f'Distribution of {col}')
        plt.show()
        sns.boxplot(quantitative_col[col])
        plt.title(f'Distribution of {col}')
        plt.show()
        print(quantitative_col[col].describe().to_markdown())
        print('\n=======================================================\n')

# =======================================================================================================
# plot_variable_pairs END
# plot_variable_pairs TO plot_categorical_and_continuous_vars
# plot_categorical_and_continuous_vars START
# =======================================================================================================

def plot_categorical_and_continuous_vars(df):
    '''
    Takes in a dataframe and separates all of the columns into category or values then
    returns a bargraph of each unique combination of category and value columns
    '''
    val_col = []
    cat_col = []
    for col in df:
        if df[col].dtype == 'O':
            cat_col.append(col)
        elif (df[col].dtype == 'float64') | (df[col].dtype == 'int'):
            val_col.append(col)
    combo_col = list(itertools.product(cat_col, val_col))
    for x, y in combo_col:
        sns.barplot(x=df[x], y=df[y])
        plt.title(f'Relationship of {x} and {y}')
        plt.axhline(y=df[y].mean(), color='r', label=f'Mean: {round(df[y].mean(), 2)}')
        plt.legend()
        plt.show()
        print(df.groupby(x)[y].describe().T.to_markdown())
        print('\n=======================================================\n')

# =======================================================================================================
# plot_categorical_and_continuous_vars END
# =======================================================================================================