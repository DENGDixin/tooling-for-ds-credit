"""
Feature engineering utilities for German Credit Dataset
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureAdder(BaseEstimator, TransformerMixin):
    """
    Custom transformer to add engineered features to the dataset.
    
    This class creates additional features that can improve model performance:
    - monthly_Payment: Ratio of LoanAmount to LoanDuration
    - jobtime_foreign: Concatenation of EmploymentDuration and ForeignWorker
    - amount_installment: Product of LoanAmount and InstallmentPercent
    - ForeignWorker_amount_Scaled: LoanAmount scaled by ForeignWorker status
    """
    
    def __init__(
            self,
            add_monthly_Payment=True,
            add_jobtime_foreign=True,
            add_amount_installment=True,
            add_amount_foreign=False
            ):
        """
        Initialize the FeatureAdder transformer.
        
        Args:
            add_monthly_Payment (bool): Whether to add monthly payment feature
            add_jobtime_foreign (bool): Whether to add job-time-foreign interaction feature
            add_amount_installment (bool): Whether to add amount-installment interaction feature
            add_amount_foreign (bool): Whether to add foreign-worker-scaled amount feature
        """
        self.add_monthly_Payment = add_monthly_Payment
        self.add_jobtime_foreign = add_jobtime_foreign
        self.add_amount_installment = add_amount_installment
        self.add_amount_foreign = add_amount_foreign

    def fit(self, X, y=None):
        """
        Fit the transformer (no actual fitting needed for this transformer).
        
        Args:
            X: Feature matrix
            y: Target vector (ignored)
            
        Returns:
            self: The fitted transformer
        """
        return self
    
    def transform(self, X):
        """
        Transform the input data by adding engineered features.
        
        Args:
            X (pd.DataFrame): Input feature matrix
            
        Returns:
            pd.DataFrame: Transformed feature matrix with additional features
        """
        X_transformed = X.copy()
        
        if self.add_monthly_Payment:
            # Create monthly payment feature: LoanAmount / LoanDuration
            # Using iloc for positional access (column 4 = LoanAmount, column 1 = LoanDuration)
            X_transformed['monthly_Payment'] = X_transformed.iloc[:, 4] / X_transformed.iloc[:, 1]
        
        if self.add_jobtime_foreign:
            # Create job-time-foreign interaction feature
            # Concatenate EmploymentDuration (column 6) and ForeignWorker (column 19)
            X_transformed['jobtime_foreign'] = X_transformed.iloc[:, 6].str.cat(
                X_transformed.iloc[:, 19], sep='_'
            )
            X_transformed['jobtime_foreign'] = X_transformed['jobtime_foreign'].astype('category')
        
        if self.add_amount_installment:
            # Create amount-installment interaction feature
            X_transformed['amount_installment'] = (
                X_transformed['LoanAmount'] * X_transformed['InstallmentPercent']
            )
        
        if self.add_amount_foreign:
            # Create foreign-worker-scaled amount feature
            # Scale LoanAmount based on ForeignWorker status (add 10% if foreign worker)
            foreign_multiplier = 1 + 0.1 * X_transformed['ForeignWorker'].map(
                {'yes': 1, 'no': 0}
            ).astype(int)
            X_transformed['ForeignWorker_amount_Scaled'] = (
                X_transformed['LoanAmount'] * foreign_multiplier
            )
        
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.
        
        Args:
            input_features: Input feature names (ignored, uses self-contained logic)
            
        Returns:
            list: Output feature names including engineered features
        """
        feature_names = []
        
        if input_features is not None:
            feature_names.extend(input_features)
        
        if self.add_monthly_Payment:
            feature_names.append('monthly_Payment')
        
        if self.add_jobtime_foreign:
            feature_names.append('jobtime_foreign')
        
        if self.add_amount_installment:
            feature_names.append('amount_installment')
        
        if self.add_amount_foreign:
            feature_names.append('ForeignWorker_amount_Scaled')
        
        return feature_names