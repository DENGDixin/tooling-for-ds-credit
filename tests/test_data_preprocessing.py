"""
Unit tests for data_preprocessing.py module using pytest
"""

import pytest
import pandas as pd
import numpy as np
import warnings
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.data_preprocessing import apply_feature_transformations


class TestApplyFeatureTransformations:
    """Test class for apply_feature_transformations function"""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data with realistic values"""
        return pd.DataFrame({
            'LoanAmount': [1000, 2000, 1500, 3000, 2500, 1800],
            'LoanDuration': [12, 24, 18, 36, 30, 20],
            'Age': [25, 35, 45, 55, 40, 30],
            'CreditHistory': ['Good', 'Bad', 'Good', 'Excellent', 'Good', 'Bad'],
            'Purpose': ['Car', 'House', 'Education', 'Business', 'Car', 'House'],
            'Employment': ['Skilled', 'Unskilled', 'Skilled', 'Management', 'Skilled', 'Unskilled']
        })
    
    @pytest.fixture
    def sample_test_data(self):
        """Create sample test data with realistic values"""
        return pd.DataFrame({
            'LoanAmount': [1200, 1800, 2500, 3500],
            'LoanDuration': [15, 20, 30, 40],
            'Age': [28, 38, 48, 58],
            'CreditHistory': ['Good', 'Bad', 'Good', 'Excellent'],
            'Purpose': ['Car', 'House', 'Education', 'Business'],
            'Employment': ['Skilled', 'Unskilled', 'Management', 'Skilled']
        })
    
    def test_single_dataset_transformation(self, sample_training_data):
        """Test transformation with only training data"""
        X = sample_training_data.copy()
        original_X = X.copy()
        
        X_transformed = apply_feature_transformations(X)
        
        # Check return type
        assert isinstance(X_transformed, pd.DataFrame)
        
        # Check that original data is unchanged
        pd.testing.assert_frame_equal(X, original_X)
        
        # Check that shape is preserved
        assert X_transformed.shape == X.shape
        
        # Check that all columns are preserved
        assert list(X_transformed.columns) == list(X.columns)
    
    def test_both_datasets_transformation(self, sample_training_data, sample_test_data):
        """Test transformation with both training and test data"""
        X = sample_training_data.copy()
        test_X = sample_test_data.copy()
        
        X_transformed, test_X_transformed = apply_feature_transformations(X, test_X)
        
        # Check return types
        assert isinstance(X_transformed, pd.DataFrame)
        assert isinstance(test_X_transformed, pd.DataFrame)
        
        # Check shapes are preserved
        assert X_transformed.shape == X.shape
        assert test_X_transformed.shape == test_X.shape
        
        # Check columns are preserved
        assert list(X_transformed.columns) == list(X.columns)
        assert list(test_X_transformed.columns) == list(test_X.columns)
    
    def test_loan_amount_transformation(self, sample_training_data):
        """Test that LoanAmount is square root transformed"""
        X = sample_training_data.copy()
        original_loan_amounts = X['LoanAmount'].copy()
        
        X_transformed = apply_feature_transformations(X)
        
        # Check square root transformation
        expected_transformed = original_loan_amounts ** 0.5
        pd.testing.assert_series_equal(
            X_transformed['LoanAmount'], 
            expected_transformed, 
            check_names=False
        )
        
        # Verify specific values
        assert X_transformed['LoanAmount'].iloc[0] == pytest.approx(np.sqrt(1000), rel=1e-9)
        assert X_transformed['LoanAmount'].iloc[1] == pytest.approx(np.sqrt(2000), rel=1e-9)
    
    def test_age_transformation(self, sample_training_data):
        """Test that Age is log transformed"""
        X = sample_training_data.copy()
        original_ages = X['Age'].copy()
        
        X_transformed = apply_feature_transformations(X)
        
        # Check log transformation
        expected_transformed = np.log(original_ages)
        pd.testing.assert_series_equal(
            X_transformed['Age'], 
            expected_transformed, 
            check_names=False
        )
        
        # Verify specific values
        assert X_transformed['Age'].iloc[0] == pytest.approx(np.log(25), rel=1e-9)
        assert X_transformed['Age'].iloc[1] == pytest.approx(np.log(35), rel=1e-9)
    
    def test_loan_duration_transformation(self, sample_training_data):
        """Test that LoanDuration is square root transformed"""
        X = sample_training_data.copy()
        original_durations = X['LoanDuration'].copy()
        
        X_transformed = apply_feature_transformations(X)
        
        # Check square root transformation
        expected_transformed = original_durations ** 0.5
        pd.testing.assert_series_equal(
            X_transformed['LoanDuration'], 
            expected_transformed, 
            check_names=False
        )
        
        # Verify specific values
        assert X_transformed['LoanDuration'].iloc[0] == pytest.approx(np.sqrt(12), rel=1e-9)
        assert X_transformed['LoanDuration'].iloc[1] == pytest.approx(np.sqrt(24), rel=1e-9)
    
    def test_categorical_columns_unchanged(self, sample_training_data):
        """Test that categorical columns are not transformed"""
        X = sample_training_data.copy()
        categorical_columns = ['CreditHistory', 'Purpose', 'Employment']
        
        # Store original categorical data
        original_categorical = {}
        for col in categorical_columns:
            original_categorical[col] = X[col].copy()
        
        X_transformed = apply_feature_transformations(X)
        
        # Check that categorical columns are unchanged
        for col in categorical_columns:
            pd.testing.assert_series_equal(
                X_transformed[col], 
                original_categorical[col], 
                check_names=False
            )
    
    def test_transformations_consistency_between_datasets(self, sample_training_data, sample_test_data):
        """Test that transformations are applied consistently to both datasets"""
        X = sample_training_data.copy()
        test_X = sample_test_data.copy()
        
        X_transformed, test_X_transformed = apply_feature_transformations(X, test_X)
        
        # Test LoanAmount transformation consistency
        assert X_transformed['LoanAmount'].iloc[0] == pytest.approx(np.sqrt(1000), rel=1e-9)
        assert test_X_transformed['LoanAmount'].iloc[0] == pytest.approx(np.sqrt(1200), rel=1e-9)
        
        # Test Age transformation consistency
        assert X_transformed['Age'].iloc[0] == pytest.approx(np.log(25), rel=1e-9)
        assert test_X_transformed['Age'].iloc[0] == pytest.approx(np.log(28), rel=1e-9)
        
        # Test LoanDuration transformation consistency
        assert X_transformed['LoanDuration'].iloc[0] == pytest.approx(np.sqrt(12), rel=1e-9)
        assert test_X_transformed['LoanDuration'].iloc[0] == pytest.approx(np.sqrt(15), rel=1e-9)
    
    def test_zero_values_handling(self):
        """Test handling of edge cases like zero values"""
        df = pd.DataFrame({
            'LoanAmount': [0, 1000, 2000],  # Zero loan amount
            'LoanDuration': [0, 12, 24],    # Zero duration
            'Age': [1, 25, 35],             # Age of 1 (edge case for log)
        })
        
        result = apply_feature_transformations(df)
        
        # Check transformations
        assert result['LoanAmount'].iloc[0] == 0  # sqrt(0) = 0
        assert result['LoanDuration'].iloc[0] == 0  # sqrt(0) = 0
        assert result['Age'].iloc[0] == pytest.approx(np.log(1), rel=1e-9)  # log(1) = 0
    
    def test_negative_values_handling(self):
        """Test handling of negative values (which shouldn't occur in real data)"""
        df = pd.DataFrame({
            'LoanAmount': [-100, 1000],
            'LoanDuration': [12, 24],
            'Age': [25, 35],
        })
        
        # This should produce NaN for square root of negative number
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = apply_feature_transformations(df)
        
        # Check that negative values produce NaN in square root
        assert np.isnan(result['LoanAmount'].iloc[0])
        assert not np.isnan(result['LoanAmount'].iloc[1])
    
    def test_large_values_handling(self):
        """Test handling of very large values"""
        df = pd.DataFrame({
            'LoanAmount': [1e6, 1e8],      # Very large loan amounts
            'LoanDuration': [100, 200],     # Long durations
            'Age': [100, 120],              # Very old ages
        })
        
        result = apply_feature_transformations(df)
        
        # Check that transformations work for large values
        assert result['LoanAmount'].iloc[0] == pytest.approx(np.sqrt(1e6), rel=1e-9)
        assert result['Age'].iloc[0] == pytest.approx(np.log(100), rel=1e-9)
        assert result['LoanDuration'].iloc[0] == pytest.approx(np.sqrt(100), rel=1e-9)
    
    def test_original_data_preservation(self, sample_training_data, sample_test_data):
        """Test that original DataFrames are not modified"""
        X = sample_training_data.copy()
        test_X = sample_test_data.copy()
        
        # Store original data
        original_X = X.copy()
        original_test_X = test_X.copy()
        
        # Apply transformations
        X_transformed, test_X_transformed = apply_feature_transformations(X, test_X)
        
        # Check that original data is unchanged
        pd.testing.assert_frame_equal(X, original_X)
        pd.testing.assert_frame_equal(test_X, original_test_X)
    
    def test_data_types_preservation(self, sample_training_data):
        """Test that non-transformed column data types are preserved"""
        X = sample_training_data.copy()
        
        # Convert categorical columns to category dtype
        categorical_cols = ['CreditHistory', 'Purpose', 'Employment']
        for col in categorical_cols:
            X[col] = X[col].astype('category')
        
        original_dtypes = X.dtypes.copy()
        
        X_transformed = apply_feature_transformations(X)
        
        # Check that categorical columns maintain their dtype
        for col in categorical_cols:
            assert X_transformed[col].dtype == original_dtypes[col]
        
        # Check that numerical transformed columns are float
        for col in ['LoanAmount', 'Age', 'LoanDuration']:
            assert np.issubdtype(X_transformed[col].dtype, np.floating)


class TestApplyFeatureTransformationsEdgeCases:
    """Test edge cases and error conditions for apply_feature_transformations"""
    
    def test_single_row_dataframe(self):
        """Test transformation with single row DataFrame"""
        df = pd.DataFrame({
            'LoanAmount': [1000],
            'LoanDuration': [12],
            'Age': [25],
            'CreditHistory': ['Good']
        })
        
        result = apply_feature_transformations(df)
        
        assert len(result) == 1
        assert result['LoanAmount'].iloc[0] == pytest.approx(np.sqrt(1000), rel=1e-9)
        assert result['Age'].iloc[0] == pytest.approx(np.log(25), rel=1e-9)
        assert result['LoanDuration'].iloc[0] == pytest.approx(np.sqrt(12), rel=1e-9)
    
    def test_mixed_data_types(self):
        """Test transformation with mixed data types"""
        df = pd.DataFrame({
            'LoanAmount': [1000.5, 2000.7],  # Float values
            'LoanDuration': [12, 24],        # Integer values
            'Age': [25.3, 35.8],             # Float values
            'CreditHistory': ['Good', 'Bad'], # String values
            'IsDefault': [True, False]       # Boolean values
        })
        
        result = apply_feature_transformations(df)
        
        # Check transformations work with mixed types
        assert result['LoanAmount'].iloc[0] == pytest.approx(np.sqrt(1000.5), rel=1e-9)
        assert result['Age'].iloc[0] == pytest.approx(np.log(25.3), rel=1e-9)
        
        # Check non-transformed columns are preserved
        assert result['CreditHistory'].iloc[0] == 'Good'
        assert result['IsDefault'].iloc[0] == True
    
    def test_nan_values_handling(self):
        """Test handling of NaN values in the data"""
        df = pd.DataFrame({
            'LoanAmount': [1000, np.nan, 2000],
            'LoanDuration': [12, 24, np.nan],
            'Age': [25, 35, 45],
            'CreditHistory': ['Good', 'Bad', 'Good']
        })
        
        result = apply_feature_transformations(df)
        
        # Check that NaN values remain NaN after transformation
        assert not np.isnan(result['LoanAmount'].iloc[0])
        assert np.isnan(result['LoanAmount'].iloc[1])
        assert not np.isnan(result['LoanAmount'].iloc[2])
        
        assert not np.isnan(result['LoanDuration'].iloc[0])
        assert not np.isnan(result['LoanDuration'].iloc[1])
        assert np.isnan(result['LoanDuration'].iloc[2])


# Performance and Integration Tests
class TestDataPreprocessingIntegration:
    """Integration tests for data_preprocessing module"""
    
    def test_realistic_german_credit_data(self):
        """Test with realistic German credit dataset structure"""
        # Realistic data mimicking actual German credit dataset
        train_data = pd.DataFrame({
            'LoanAmount': [1169, 5951, 2096, 7882, 4870, 9055, 2835, 6948, 3059, 5234],
            'LoanDuration': [6, 48, 12, 42, 24, 36, 24, 36, 12, 30],
            'Age': [67, 22, 49, 45, 53, 35, 28, 39, 61, 28],
            'InstallmentPercent': [4, 2, 2, 2, 3, 2, 4, 2, 1, 4],
            'ResidenceDuration': [4, 2, 3, 4, 4, 4, 4, 2, 4, 4],
            'ExistingCredits': [2, 1, 1, 1, 2, 1, 1, 1, 1, 2],
            'Purpose': ['radio/tv', 'radio/tv', 'education', 'furniture/equipment', 
                       'car', 'education', 'furniture/equipment', 'car', 'radio/tv', 'car'],
            'CreditHistory': ['critical/other existing credit', 'existing paid', 
                             'existing paid', 'existing paid', 'delayed previously',
                             'existing paid', 'existing paid', 'existing paid', 
                             'existing paid', 'critical/other existing credit']
        })
        
        test_data = pd.DataFrame({
            'LoanAmount': [2169, 3951, 4096, 1882, 2870],
            'LoanDuration': [18, 24, 36, 12, 30],
            'Age': [43, 38, 29, 55, 41],
            'InstallmentPercent': [3, 4, 2, 1, 3],
            'ResidenceDuration': [3, 4, 2, 4, 3],
            'ExistingCredits': [1, 2, 1, 1, 2],
            'Purpose': ['car', 'furniture/equipment', 'radio/tv', 'education', 'car'],
            'CreditHistory': ['existing paid', 'existing paid', 'delayed previously', 
                             'existing paid', 'existing paid']
        })
        
        # Apply transformations
        train_transformed, test_transformed = apply_feature_transformations(train_data, test_data)
        
        # Verify transformations
        assert train_transformed.shape == train_data.shape
        assert test_transformed.shape == test_data.shape
        
        # Check specific transformations
        assert train_transformed['LoanAmount'].iloc[0] == pytest.approx(np.sqrt(1169), rel=1e-9)
        assert train_transformed['Age'].iloc[0] == pytest.approx(np.log(67), rel=1e-9)
        assert train_transformed['LoanDuration'].iloc[0] == pytest.approx(np.sqrt(6), rel=1e-9)
        
        # Check that other columns are preserved
        assert train_transformed['Purpose'].iloc[0] == 'radio/tv'
        assert train_transformed['InstallmentPercent'].iloc[0] == 4
    
    def test_memory_efficiency(self):
        """Test that transformations don't consume excessive memory"""
        # Create a moderately large dataset
        n_rows = 10000
        large_df = pd.DataFrame({
            'LoanAmount': np.random.uniform(500, 10000, n_rows),
            'LoanDuration': np.random.randint(6, 72, n_rows),
            'Age': np.random.randint(18, 80, n_rows),
            'Purpose': np.random.choice(['car', 'house', 'education'], n_rows),
            'CreditHistory': np.random.choice(['good', 'bad', 'excellent'], n_rows)
        })
        
        # Test that transformation completes without memory issues
        result = apply_feature_transformations(large_df)
        
        assert result.shape == large_df.shape
        assert len(result) == n_rows
        
        # Check that transformations were applied
        assert not result['LoanAmount'].equals(large_df['LoanAmount'])
        assert not result['Age'].equals(large_df['Age'])
        assert not result['LoanDuration'].equals(large_df['LoanDuration'])
        
        # Check that categorical columns are preserved
        assert result['Purpose'].equals(large_df['Purpose'])
        assert result['CreditHistory'].equals(large_df['CreditHistory'])