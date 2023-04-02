# =======================================================================================================
# Table of Contents START
# =======================================================================================================

'''
1. Orientation
2. Imports
3. get_eval_stats
4. plot_residuals
5. regression_errors
6. baseline_mean_errors
7. better_than_baseline
'''

# =======================================================================================================
# Table of Contents END
# Table of Contents TO Orientation
# Orientation START
# =======================================================================================================

'''
The purpose of this file is to create functions for ease of use when conducting evaluations on
regression type machine learning models.
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
from sklearn.metrics import r2_score

# Local .py files
import wrangle

# =======================================================================================================
# Imports END
# Imports TO get_eval_stats
# get_eval_stats START
# =======================================================================================================

def get_eval_stats(df, actual_col, baseline_col, prediction_col):
    '''
    Takes in 4 inputs and returns summary evaluation statistics for the baseline, the prediction,
    and the summary evaluation of the difference between the two...

    INPUT VALUES:
    df = Pandas dataframe name
    actual_col = Column name containing actual values
    baseline_col = Column name containing baseline prediction of actual_col
    prediction_col = Column name containing predicted values of actual_col

    OUTPUT VALUES:
    3 Tables with summary statistics for the baseline, prediction, and difference
    (Stats like SSE, ESS, TSS, MSE, RMSE, R2, etc.)
    '''
    baseline = df[actual_col].mean()
    base_residual = df[actual_col] - baseline
    pred_residual = df[actual_col] - df[prediction_col]
    SSE_base = (base_residual ** 2).sum()
    SSE_pred = (pred_residual ** 2).sum()
    SSE_diff = int(SSE_pred - SSE_base)
    ESS_base = sum((df[baseline_col] - df[actual_col]) ** 2)
    ESS_pred = sum((df[prediction_col] - df[actual_col]) ** 2)
    ESS_diff = int(ESS_pred - ESS_base)
    TSS_base = SSE_base + ESS_base
    TSS_pred = SSE_pred + ESS_pred
    TSS_diff = int(TSS_pred - TSS_base)
    MSE_base = SSE_base / len(df)
    MSE_pred = SSE_pred / len(df)
    MSE_diff = int(MSE_pred - MSE_base)
    RMSE_base = MSE_base ** .5
    RMSE_pred = MSE_pred ** .5
    RMSE_diff = int(RMSE_pred - RMSE_base)
    R2_base = 1 - (SSE_base/TSS_base)
    R2_pred = 1 - (SSE_pred/SSE_base)
    R2_diff = R2_pred - R2_base
    print(f'\033[35m===== {baseline_col} =====\033[0m\n\033[32mSSE:\033[0m {SSE_base:.2f}\n\033[32mESS:\033[0m {ESS_base:.2f}\n\033[32mTSS:\033[0m {TSS_base:.2f}\n\033[32mMSE:\033[0m {MSE_base:.2f}\n\033[32mRMSE:\033[0m {RMSE_base:.2f}\n')
    print(f'\033[35m===== {prediction_col} =====\033[0m\n\033[32mSSE:\033[0m {SSE_pred:.2f}\n\033[32mESS:\033[0m {ESS_pred:.2f}\n\033[32mTSS:\033[0m {TSS_pred:.2f}\n\033[32mMSE:\033[0m {MSE_pred:.2f}\n\033[32mRMSE:\033[0m {RMSE_pred:.2f}\n\033[32mR2:\033[0m {R2_pred:.2f}\n')
    print(f'\033[35m===== {prediction_col} - {baseline_col} =====\033[0m\n\033[32mSSE:\033[0m {SSE_diff:.2f}\n\033[32mESS:\033[0m {ESS_diff:.2f}\n\033[32mTSS:\033[0m {TSS_diff:.2f}\n\033[32mMSE:\033[0m {MSE_diff:.2f}\n\033[32mRMSE:\033[0m {RMSE_diff:.2f}\n')

# =======================================================================================================
# get_eval_stats END
# get_eval_stats TO plot_residuals
# plot_residuals START
# =======================================================================================================

def plot_residuals(y, yhat):
    '''
    Takes the input of the actual and prediction columns in a pandas dataframe and outputs a visualization.

    INPUT:
    y = pandas_df.actual_values
    yhat = pandas_df.predicted_values

    OUTPUT:
    Scatter plot of residuals and a reference line
    '''
    residuals = y - yhat
    sns.scatterplot(x=yhat, y=residuals)
    plt.axhline(y=0, color='r', label='Actual Line')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.legend()
    plt.show()

# =======================================================================================================
# plot_residuals END
# plot_residuals TO regression_errors
# regression_errors START
# =======================================================================================================

def regression_errors(y, yhat):
    '''
    Takes the input of the actual and prediction columns in a pandas dataframe and outputs
    summary stats for the prediction column.

    INPUT:
    y = pandas_df.actual_values
    yhat = pandas_df.predicted_values

    OUTPUT:
    Stats like SSE, ESS, TSS, MSE, RMSE
    '''
    residual = y - yhat
    SSE = sum((residual ** 2))
    ESS = sum((yhat - y) ** 2)
    TSS = SSE + ESS
    MSE = SSE / len(y)
    RMSE = MSE ** .5
    print(f'\033[32mSSE:\033[0m {SSE:.2f}\n\033[32mESS:\033[0m {ESS:.2f}\n\033[32mTSS:\033[0m {TSS:.2f}\n\033[32mMSE:\033[0m {MSE:.2f}\n\033[32mRMSE:\033[0m {RMSE:.2f}\n')

# =======================================================================================================
# regression_errors END
# regression_errors TO baseline_mean_errors
# baseline_mean_errors START
# =======================================================================================================

def baseline_mean_errors(y):
    '''
    Takes the input of the actual column in a pandas dataframe and outputs summary stats for the 
    baseline of the actual column.

    INPUT:
    y = pandas_df.actual_values

    OUTPUT:
    Stats like SSE, MSE, RMSE
    '''
    baseline = y.mean()
    residual = y - baseline
    SSE = sum((residual ** 2))
    MSE = SSE / len(y)
    RMSE = MSE ** .5
    print(f'\033[32mSSE:\033[0m {SSE:.2f}\n\033[32mMSE:\033[0m {MSE:.2f}\n\033[32mRMSE:\033[0m {RMSE:.2f}\n')

# =======================================================================================================
# baseline_mean_errors END
# baseline_mean_errors TO better_than_baseline
# better_than_baseline START
# =======================================================================================================

def better_than_baseline(y, baseline, yhat):
    '''
    Takes the input of the actual, baseline, and prediction columns in a pandas dataframe and outputs
    whether or not the prediction is better than the baseline as well as their r2 scores

    INPUT:
    y = pandas_df.actual_values
    baseline = pandas_df.baseline_values
    yhat = pandas_df.predicted_values

    OUTPUT:
    Yes/No of prediction performance vs. baseline and the difference between the two as well as
    the baseline and prediction r2 scores
    '''
    base_r2 = r2_score(y, baseline)
    pred_r2 = r2_score(y, yhat)
    if pred_r2 > base_r2:
        print(f'\033[32mBETTER THAN BASELINE!\033[0m: {(pred_r2 - base_r2):.4f}')
        print(f'Baseline: {base_r2:.4f}')
        print(f'Prediction: {pred_r2:.4f}')
    else:
        print(f'\00[31mWORSE THAN BASELINE!\033[0m: {(pred_r2 - base_r2):.4f}')
        print(f'Baseline: {base_r2:.4f}')
        print(f'Prediction: {pred_r2:.4f}')

# =======================================================================================================
# better_than_baseline END
# =======================================================================================================