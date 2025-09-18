"""
Best performing models and analysis pipelines for German Credit Dataset
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer

from src.utils.feature_engineering import FeatureAdder
from src.model.model_configs import BEST_MODEL_PARAMS


def create_preprocessor(X):
    """
    Create preprocessor for the best models.
    
    Args:
        X (pd.DataFrame): Feature matrix to determine column types
        
    Returns:
        ColumnTransformer: Configured preprocessor
    """
    cat_columns = X.select_dtypes(include='category').columns.drop(['ForeignWorker'])
    num_columns = X.select_dtypes(include='number').columns
    num_columns = num_columns.union(pd.Index(['monthly_Payment', 'amount_installment', 'ForeignWorker_amount_Scaled']))

    onehot = OneHotEncoder(handle_unknown='ignore')
    scaler = StandardScaler()

    preprocessor = ColumnTransformer([
        ('onehot', onehot, cat_columns),
        ('scaler', scaler, num_columns)
    ])
    
    return preprocessor


def create_pipxgb_for_analysis(X):
    """
    Create XGBoost pipeline for feature importance analysis using optimized parameters.
    
    Args:
        X (pd.DataFrame): Feature matrix
        
    Returns:
        Pipeline: XGBoost pipeline configured for analysis
    """
    preprocessor = create_preprocessor(X)
    
    feature_adder = FeatureAdder(
        add_monthly_Payment=True,
        add_amount_installment=True,
        add_jobtime_foreign=True,
        add_amount_foreign=True
    )

    pipxgb = make_pipeline(
        feature_adder,
        preprocessor,
        xgb.XGBClassifier(**BEST_MODEL_PARAMS['xgb'])
    )
    
    return pipxgb


def create_best_models(X):
    """
    Create the best performing models with optimized hyperparameters.
    
    Args:
        X (pd.DataFrame): Feature matrix
        
    Returns:
        dict: Dictionary containing best model pipelines
    """
    preprocessor = create_preprocessor(X)
    
    feature_adder = FeatureAdder(add_amount_foreign=True)
    
    # Best XGBoost model
    best_model_xgb = make_pipeline(
        feature_adder,
        preprocessor,
        xgb.XGBClassifier(**BEST_MODEL_PARAMS['xgb'])
    )
    
    # Best SVC model
    best_model_svc = make_pipeline(
        feature_adder,
        preprocessor,
        SVC(**BEST_MODEL_PARAMS['svc'])
    )
    
    return {
        'xgb': best_model_xgb,
        'svc': best_model_svc
    }


def create_analysis_pipeline(X):
    """
    Create pipeline specifically for SHAP analysis and feature importance.
    
    Args:
        X (pd.DataFrame): Feature matrix
        
    Returns:
        Pipeline: Analysis pipeline
    """
    return create_pipxgb_for_analysis(X)