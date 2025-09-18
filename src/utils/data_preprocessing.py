"""
Data preprocessing utilities for German Credit Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def apply_feature_transformations(X, test_X=None):
    """
    Apply feature transformations to the datasets.
    
    Args:
        X (pd.DataFrame): Training features
        test_X (pd.DataFrame, optional): Test features
        
    Returns:
        tuple: (X_transformed, test_X_transformed) if test_X provided, else X_transformed
    """
    X_transformed = X.copy()
    
    # Apply square root transformation to LoanAmount
    X_transformed['LoanAmount'] = X_transformed['LoanAmount'] ** 0.5
    
    # Apply log transformation to Age
    X_transformed['Age'] = np.log(X_transformed['Age'])
    
    # Apply square root transformation to LoanDuration
    X_transformed['LoanDuration'] = X_transformed['LoanDuration'] ** 0.5
    
    if test_X is not None:
        test_X_transformed = test_X.copy()
        test_X_transformed['LoanAmount'] = test_X_transformed['LoanAmount'] ** 0.5
        test_X_transformed['Age'] = np.log(test_X_transformed['Age'])
        test_X_transformed['LoanDuration'] = test_X_transformed['LoanDuration'] ** 0.5
        return X_transformed, test_X_transformed
    
    return X_transformed
