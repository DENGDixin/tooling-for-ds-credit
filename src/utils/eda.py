"""
Utility functions for exploratory data analysis (EDA).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_feature_distribution(data, feature_name, bins=40):
    """
    Plot histogram of a feature.
    
    Args:
        data (pd.DataFrame or pd.Series): Data containing the feature
        feature_name (str): Name of the feature to plot
        bins (int): Number of bins for histogram
    """
    if isinstance(data, pd.DataFrame):
        data[feature_name].hist(bins=bins)
    else:
        data.hist(bins=bins)
    
    plt.title(f'Distribution of {feature_name}')
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.show()


def get_data_info(data, name="Dataset"):
    """
    Display basic information about the dataset.
    
    Args:
        data (pd.DataFrame): Dataset to analyze
        name (str): Name for display purposes
    """
    print(f"\n{name} Info:")
    print(f"Shape: {data.shape}")
    print(f"\nFirst 5 rows:")
    print(data.head())
    print(f"\nData types and info:")
    data.info()
    print(f"\nDescriptive statistics:")
    print(data.describe())


def quick_eda(trainset, testset):
    """
    Perform quick exploratory data analysis.
    
    Args:
        trainset (pd.DataFrame): Training dataset
        testset (pd.DataFrame): Test dataset
    """
    print("=== German Credit Dataset - Quick EDA ===")
    
    # Basic info
    get_data_info(trainset, "Training Set")
    get_data_info(testset, "Test Set")
    
    # Plot key features before transformation
    print("\nPlotting original feature distributions...")
    plot_feature_distribution(trainset, 'LoanAmount')
    plot_feature_distribution(trainset, 'Age')
    plot_feature_distribution(trainset, 'LoanDuration')
    
    # Apply transformations
    trainset_transformed = trainset.copy()
    trainset_transformed['LoanAmount'] = trainset_transformed['LoanAmount'] ** 0.5
    trainset_transformed['Age'] = np.log(trainset_transformed['Age'])
    trainset_transformed['LoanDuration'] = trainset_transformed['LoanDuration'] ** 0.5
    
    # Plot transformed features
    print("\nPlotting transformed feature distributions...")
    plot_feature_distribution(trainset_transformed, 'LoanAmount')
    plot_feature_distribution(trainset_transformed, 'Age')
    plot_feature_distribution(trainset_transformed, 'LoanDuration')