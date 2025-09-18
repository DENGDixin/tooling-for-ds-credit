"""
Data loading utilities for German Credit Dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_german_credit_data(data_dir="data"):
    """
    Load the German credit training and test datasets.
    
    Args:
        data_dir (str): Directory containing the data files
        
    Returns:
        tuple: (trainset, testset) as pandas DataFrames
    """
    data_path = Path(data_dir)
    
    trainset = pd.read_csv(data_path / 'german_credit_train.csv')
    testset = pd.read_csv(data_path / 'german_credit_test.csv')
    
    return trainset, testset


def prepare_features_and_target(trainset, testset):
    """
    Prepare features and target from the datasets.
    
    Args:
        trainset (pd.DataFrame): Training dataset
        testset (pd.DataFrame): Test dataset
        
    Returns:
        tuple: (X, y, test_X) where X is training features, y is target, test_X is test features
    """
    # Convert Risk column to binary
    trainset_copy = trainset.copy()
    trainset_copy['Risk'] = trainset_copy['Risk'].map({'Risk': 1, 'No Risk': 0})
    
    # Split features and target
    X = trainset_copy.drop(columns='Risk')
    y = trainset_copy['Risk']
    
    # Test features (remove Id column)
    test_X = testset.drop(columns='Id')
    
    # Convert categorical columns
    cat_columns = X.select_dtypes(include='object').columns
    X[cat_columns] = X[cat_columns].astype('category')
    test_X[cat_columns] = test_X[cat_columns].astype('category')
    
    return X, y, test_X