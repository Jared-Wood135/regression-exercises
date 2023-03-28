# =======================================================================================================
# Table of Contents START
# =======================================================================================================

'''
1. Orientation
2. Imports
3. acquire
4. prepare
5. wrangle_zillow
6. split
7. remove_outliers
'''

# =======================================================================================================
# Table of Contents END
# Table of Contents TO Orientation
# Orientation START
# =======================================================================================================

'''
The purpose of this file is to create functions for both the acquire & preparation phase of the data
science pipeline or also known as 'wrangle' data...
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
from scipy import stats
from sklearn.model_selection import train_test_split
import os
import env

# =======================================================================================================
# Imports END
# Imports TO acquire
# acquire START
# =======================================================================================================

def acquire():
    '''
    Obtains the vanilla version of zillow dataframe
    '''
    query = '''
    SELECT 
        bedroomcnt,
        bathroomcnt,
        calculatedfinishedsquarefeet,
        taxvaluedollarcnt,
        yearbuilt,
        taxamount,
        fips,
        propertylandusedesc
    FROM 
        properties_2017
        LEFT JOIN propertylandusetype USING(propertylandusetypeid)
    WHERE
        propertylandusetypeid = 261 OR propertylandusetypeid = 279'''
    url = env.get_db_url('zillow')
    zillow = pd.read_sql(query, url)
    return zillow

# =======================================================================================================
# acquire END
# acquire TO prepare
# prepare START
# =======================================================================================================

def prepare():
    '''
    Takes in the vanilla zillow dataframe and returns a cleaned version that is ready for exploration
    and further analysis
    '''
    zillow = acquire()
    zillow.bedroomcnt = zillow.bedroomcnt.fillna(3.0)
    zillow.bedroomcnt = zillow.bedroomcnt.astype(int)
    zillow.bathroomcnt = zillow.bathroomcnt.fillna(2.0)
    zillow.calculatedfinishedsquarefeet = zillow.calculatedfinishedsquarefeet.fillna(1862.9)
    zillow.taxvaluedollarcnt = zillow.taxvaluedollarcnt.fillna(461896.2)
    zillow.yearbuilt = zillow.yearbuilt.fillna(1955)
    zillow.yearbuilt = zillow.yearbuilt.astype(int)
    zillow.taxamount = zillow.taxamount.fillna(5634.87)
    zillow['state'] = 'California'
    zillow = zillow.rename(columns={'fips' : 'county', 
                                    'calculatedfinishedsquarefeet' : 'sqrft', 
                                    'taxvaluedollarcnt' : 'assessedvalue',
                                    })
    zillow.county = np.where(zillow.county == 6037, 'Los Angeles', zillow.county)
    zillow.county = np.where(zillow.county == '6059.0', 'Orange', zillow.county)
    zillow.county = np.where(zillow.county == '6111.0', 'Ventura', zillow.county)
    dummies = pd.get_dummies(zillow.drop(columns='state').select_dtypes(include='object'))
    zillow = pd.concat([zillow, dummies], axis=1)
    outlier_cols = ['bedroomcnt',
                    'bathroomcnt',
                    'sqrft',
                    'assessedvalue',
                    'taxamount']
    zillow = remove_outliers(zillow, outlier_cols)
    return zillow

# =======================================================================================================
# prepare END
# prepare TO wrangle_zillow
# wrangle_zillow START
# =======================================================================================================

def wrangle_zillow():
    '''
    Function that acquires and prepares the zillow dataframe for use as well as creating a csv.
    '''
    if os.path.exists('zillow.csv'):
        return pd.read_csv('zillow.csv', index_col=0)
    else:
        zillow = prepare()
        zillow.to_csv('zillow.csv')
        return pd.read_csv('zillow.csv', index_col=0)
    
# =======================================================================================================
# wrangle_zillow END
# wrangle_zillow TO split
# split START
# =======================================================================================================

def split(df):
    '''
    Takes a dataframe and splits the data into a train, validate and test datasets
    '''
    train_val, test = train_test_split(df, train_size=0.8, random_state=1349)
    train, validate = train_test_split(train_val, train_size=0.7, random_state=1349)
    print(f"train.shape:{train.shape}\nvalidate.shape:{validate.shape}\ntest.shape:{test.shape}")
    return train, validate, test


# =======================================================================================================
# split END
# split TO remove_outliers
# remove_outliers START
# =======================================================================================================

def remove_outliers(df, col_list, k=1.5):
    '''
    remove outliers from a dataframe based on a list of columns using the tukey method
    returns a single dataframe with outliers removed
    '''
    col_qs = {}
    for col in col_list:
        col_qs[col] = q1, q3 = df[col].quantile([0.25, 0.75])
    for col in col_list:
        iqr = col_qs[col][0.75] - col_qs[col][0.25]
        lower_fence = col_qs[col][0.25] - (k*iqr)
        upper_fence = col_qs[col][0.75] + (k*iqr)
        df = df[(df[col] > lower_fence) & (df[col] < upper_fence)]
    return df

# =======================================================================================================
# remove_outliers END
# =======================================================================================================